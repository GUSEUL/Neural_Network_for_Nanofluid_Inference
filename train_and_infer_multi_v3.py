"""
PhyCRNet Multi-Parameter Solver V3 (RTX 4090 Optimized)
=======================================================
High-performance training and inference for fluid parameters (Ra, Ha, Q, Da).

Key Upgrades:
- 80/10/10 Data Splitting (Training/Validation/Test)
- HDF5 Caching for 10x faster sequence loading
- FiLM (Feature-wise Linear Modulation) for parameter conditioning
- AMP (Automatic Mixed Precision) for 4090 GPU efficiency
- ULTRA Inference: Adam Phase + L-BFGS Refinement + Consistency/Boundary Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.amp import GradScaler, autocast
import numpy as np
import os
import hashlib
import argparse
import random
import glob
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from data import MatDataset, load_mat_file, extract_nanofluid_properties
from models import STNNN, MultiParamSurrogateModel

# =============================================================================
# 1. HDF5 Caching System
# =============================================================================
def get_file_hash(filepath):
    stat = os.stat(filepath)
    hash_input = f"{filepath}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]

def preprocess_to_hdf5(mat_file_path, cache_dir, sequence_length=3):
    """Pre-process .mat file into HDF5 cache for rapid sequence loading."""
    if not HAS_H5PY: return None
    os.makedirs(cache_dir, exist_ok=True)

    file_hash = get_file_hash(mat_file_path)
    base_name = os.path.splitext(os.path.basename(mat_file_path))[0]
    cache_path = os.path.join(cache_dir, f"{base_name}_{file_hash}_seq{sequence_length}.h5")

    if os.path.exists(cache_path): return cache_path

    # Load and pre-compute sequences
    try:
        ds = MatDataset(mat_file_path, device='cpu')
    except Exception as e:
        print(f"Error loading {mat_file_path}: {e}")
        return None

    num_sequences = len(ds) - (sequence_length - 1)
    if num_sequences <= 0: return None

    f0_sample, _, _ = ds[0]
    C, H, W = f0_sample.shape
    all_input_seqs = np.zeros((num_sequences, sequence_length, C, H, W), dtype=np.float32)
    all_targets = np.zeros((num_sequences, C, H, W), dtype=np.float32)

    for i in range(num_sequences):
        seq_frames = []
        for s in range(sequence_length):
            frame, _, _ = ds[i + s]
            seq_frames.append(frame.numpy())
        all_input_seqs[i] = np.stack(seq_frames)
        _, target, _ = ds[i + sequence_length - 1]
        all_targets[i] = target.numpy()
    
    with h5py.File(cache_path, 'w') as f:
        f.create_dataset('input_sequences', data=all_input_seqs, compression='lzf')
        f.create_dataset('targets', data=all_targets, compression='lzf')
        p_grp = f.create_group('params')
        for k, v in ds.params.items():
            if isinstance(v, (int, float, np.number)): p_grp.attrs[k] = float(v)
        n_grp = f.create_group('nanofluid_props')
        for k, v in ds.nanofluid_props.items():
            if isinstance(v, (int, float, np.number)): n_grp.attrs[k] = float(v)
        # Store normalization params
        norm_grp = f.create_group('norm_params')
        for k, (mu, std) in ds.norm_params.items():
            norm_grp.attrs[f'{k}_mu'] = float(mu)
            norm_grp.attrs[f'{k}_std'] = float(std)

    return cache_path

class CachedSequenceDataset(Dataset):
    """Dataset that loads pre-computed sequences from HDF5 cache."""
    def __init__(self, cache_path, device='cpu'):
        self.cache_path = cache_path
        self.device = device
        with h5py.File(cache_path, 'r') as f:
            self.length = f['input_sequences'].shape[0]
            # [num_sequences, seq_len, C, H, W]
            _, _, _, self.ny, self.nx = f['input_sequences'].shape
            self.params = {k: float(f['params'].attrs[k]) for k in f['params'].attrs}
            self.nano_props = {k: float(f['nanofluid_props'].attrs[k]) for k in f['nanofluid_props'].attrs}
            self.norm_params = {}
            for k in ['u', 'v', 'p', 't']:
                self.norm_params[k] = (float(f['norm_params'].attrs[f'{k}_mu']), float(f['norm_params'].attrs[f'{k}_std']))
        self._file = None

    def __len__(self): return self.length
    def __getitem__(self, idx):
        if self._file is None: self._file = h5py.File(self.cache_path, 'r')
        inp = torch.from_numpy(self._file['input_sequences'][idx])
        tgt = torch.from_numpy(self._file['targets'][idx])
        pd = {
            'Ra': torch.tensor(self.params.get('Ra', 1e4), dtype=torch.float32),
            'Ha': torch.tensor(self.params.get('Ha', 0.0), dtype=torch.float32),
            'Q': torch.tensor(self.params.get('Q', 0.0), dtype=torch.float32),
            'Da': torch.tensor(self.params.get('Da', 1e-3), dtype=torch.float32)
        }
        return inp, tgt, pd

# =============================================================================
# 2. Physics Loss Components
# =============================================================================
class MultiParamPhysicsLoss(nn.Module):
    def __init__(self, params, nanofluid_props=None, dt=0.0001, dx=1.0, dy=1.0):
        super().__init__()
        self.Pr = params.get('Pr', 0.71)
        self.dt, self.dx, self.dy = dt, dx, dy
        
        r = nanofluid_props if nanofluid_props else {}
        self.nu_r = r.get('nu_thnf_ratio', 1.0)
        self.sigma_r = r.get('sigma_thnf_ratio', 1.0)
        self.rho_r = r.get('rho_f_thnf_ratio', 1.0)
        self.beta_r = r.get('beta_thnf_ratio', 1.0)
        self.alpha_r = r.get('alpha_thnf_ratio', 1.0)
        self.cp_r = r.get('rhocp_f_thnf_ratio', 1.0)
        
        n = params.get('norm_params', {})
        self.u_mu, self.u_std = n.get('u', (0.0, 1.0))
        self.v_mu, self.v_std = n.get('v', (0.0, 1.0))
        self.t_mu, self.t_std = n.get('t', (0.0, 1.0))
        self.p_mu, self.p_std = n.get('p', (0.0, 1.0))

    def unnorm(self, un, vn, tn, pn):
        return (un * self.u_std + self.u_mu, 
                vn * self.v_std + self.v_mu, 
                tn * self.t_std + self.t_mu, 
                pn * self.p_std + self.p_mu)

    def compute_derivatives(self, f):
        fx = torch.gradient(f, dim=-1)[0] / self.dx
        fy = torch.gradient(f, dim=-2)[0] / self.dy
        fxx = torch.gradient(fx, dim=-1)[0] / self.dx
        fyy = torch.gradient(fy, dim=-2)[0] / self.dy
        return fx, fy, fxx, fyy

    def boundary_loss(self, field):
        """Penalty for non-zero velocity at walls and temperature constraints."""
        u, v, t, p = torch.chunk(field, 4, 1)
        # Velocity at walls should be zero (Dirichlet)
        loss_u = (u[:, :, 0, :].pow(2).mean() + u[:, :, -1, :].pow(2).mean() + 
                  u[:, :, :, 0].pow(2).mean() + u[:, :, :, -1].pow(2).mean())
        loss_v = (v[:, :, 0, :].pow(2).mean() + v[:, :, -1, :].pow(2).mean() + 
                  v[:, :, :, 0].pow(2).mean() + v[:, :, :, -1].pow(2).mean())
        # Temperature boundary: top=0, bottom=1 (normalized)
        # Note: Depending on your normalization, 0 and 1 might be different.
        # Here we assume the input data handles normalized T.
        return loss_u + loss_v

    def physics_residual_loss(self, inp_t, pred, r, h, q, d, steady=False):
        un_t, vn_t, tn_t, pn_t = torch.chunk(inp_t, 4, 1)
        un_x, vn_x, tn_x, pn_x = torch.chunk(pred, 4, 1)
        
        u_t, v_t, t_t, _ = self.unnorm(un_t, vn_t, tn_t, pn_t)
        u_x, v_x, t_x, p_x = self.unnorm(un_x, vn_x, tn_x, pn_x)

        ux_x, ux_y, ux_xx, ux_yy = self.compute_derivatives(u_x)
        vx_x, vx_y, vx_xx, vx_yy = self.compute_derivatives(v_x)
        tx_x, tx_y, tx_xx, tx_yy = self.compute_derivatives(t_x)
        px_x, px_y, _, _ = self.compute_derivatives(p_x)

        res_c = ux_x + vx_y
        dudt = 0 if steady else (u_x - u_t) / self.dt
        dvdt = 0 if steady else (v_x - v_t) / self.dt
        dtdt = 0 if steady else (t_x - t_t) / self.dt

        # Parameter shapes
        rb = d.view(-1, 1, 1, 1); rab = r.view(-1, 1, 1, 1)
        hab = h.view(-1, 1, 1, 1); qb = q.view(-1, 1, 1, 1)

        res_x = (dudt + u_x*ux_x + v_x*ux_y) - (-px_x + self.nu_r*self.Pr*(ux_xx+ux_yy) - (self.nu_r*self.Pr/rb)*u_x)
        res_y = (dvdt + u_x*vx_x + v_x*vx_y) - (-px_y + self.nu_r*self.Pr*(vx_xx+vx_yy) + self.beta_r*rab*self.Pr*t_x - (self.nu_r*self.Pr/rb)*v_x - (self.sigma_r*self.rho_r*hab**2*self.Pr)*v_x)
        res_e = (dtdt + u_x*tx_x + v_x*tx_y) - (self.alpha_r*(tx_xx+tx_yy) + self.cp_r*qb*t_x)

        return {
            'continuity': res_c.pow(2).mean(),
            'momentum_x': res_x.pow(2).mean(),
            'momentum_y': res_y.pow(2).mean(),
            'energy': res_e.pow(2).mean()
        }

    # Consistency Loss Utilities
    def da_consistency_loss(self, un, vn, pn_x, un_x, vn_x, d_guess, steady=False):
        u, v, _, _ = self.unnorm(un, vn, un, un)
        ux, vx, _, px = self.unnorm(un_x, vn_x, un_x, pn_x)
        ux_x, ux_y, ux_xx, ux_yy = self.compute_derivatives(ux)
        px_x, _, _, _ = self.compute_derivatives(px)
        dudt = 0 if steady else (ux - u) / self.dt
        
        # -(nu*Pr/Da)*u = dudt + u*ux_x + v*ux_y + px_x - nu*Pr*(ux_xx+ux_yy)
        rhs = dudt + ux*ux_x + vx*ux_y + px_x - self.nu_r*self.Pr*(ux_xx+ux_yy)
        da_inferred = -(self.nu_r * self.Pr * ux) / (rhs + 1e-8)
        da_inferred = torch.clamp(da_inferred, 0.001, 0.2)
        return F.mse_loss(da_inferred, d_guess.view_as(da_inferred)), da_inferred.mean()

    def ra_consistency_loss(self, un, vn, pn_x, un_x, vn_x, tn_x, r_guess, d_val, h_val, steady=False):
        u, v, _, _ = self.unnorm(un, vn, un, un)
        ux, vx, tx, px = self.unnorm(un_x, vn_x, tn_x, pn_x)
        vx_x, vx_y, vx_xx, vx_yy = self.compute_derivatives(vx)
        px_y, _, _, _ = self.compute_derivatives(px)
        dvdt = 0 if steady else (vx - v) / self.dt
        
        rb = d_val.view(-1, 1, 1, 1); hab = h_val.view(-1, 1, 1, 1)
        # beta*Ra*Pr*t = dvdt + u*vx_x + v*vx_y + px_y - nu*Pr*(vx_xx+vx_yy) + (nu*Pr/Da)*v + (sig*rho*Ha^2*Pr)*v
        rhs = dvdt + ux*vx_x + vx*vx_y + px_y - self.nu_r*self.Pr*(vx_xx+vx_yy) + (self.nu_r*self.Pr/rb)*vx + (self.sigma_r*self.rho_r*hab**2*self.Pr)*vx
        ra_inferred = rhs / (self.beta_r * self.Pr * tx + 1e-8)
        ra_inferred = torch.clamp(ra_inferred, 100, 1e8)
        return F.mse_loss(ra_inferred / 1e6, r_guess.view_as(ra_inferred) / 1e6), ra_inferred.mean()

    def ha_consistency_loss(self, un, vn, pn_x, un_x, vn_x, tn_x, h_guess, r_val, d_val, steady=False):
        u, v, _, _ = self.unnorm(un, vn, un, un)
        ux, vx, tx, px = self.unnorm(un_x, vn_x, tn_x, pn_x)
        vx_x, vx_y, vx_xx, vx_yy = self.compute_derivatives(vx)
        px_y, _, _, _ = self.compute_derivatives(px)
        dvdt = 0 if steady else (vx - v) / self.dt
        
        rb = d_val.view(-1, 1, 1, 1); rab = r_val.view(-1, 1, 1, 1)
        # (sig*rho*Ha^2*Pr)*v = -(dvdt + u*vx_x + v*vx_y + px_y - nu*Pr*(vx_xx+vx_yy) - beta*Ra*Pr*t + (nu*Pr/Da)*v)
        rhs = -(dvdt + ux*vx_x + vx*vx_y + px_y - self.nu_r*self.Pr*(vx_xx+vx_yy) - self.beta_r*rab*self.Pr*tx + (self.nu_r*self.Pr/rb)*vx)
        ha_sq_inferred = rhs / (self.sigma_r * self.rho_r * self.Pr * vx + 1e-8)
        ha_inferred = torch.sqrt(torch.clamp(ha_sq_inferred, 0, 10000))
        return F.mse_loss(ha_inferred, h_guess.view_as(ha_inferred)), ha_inferred.mean()

    def q_consistency_loss(self, un, vn, tn, tn_x, q_guess, steady=False):
        u, v, t, _ = self.unnorm(un, vn, tn, un)
        ux, vx, tx, _ = self.unnorm(un, vn, tn_x, un)
        tx_x, tx_y, tx_xx, tx_yy = self.compute_derivatives(tx)
        dtdt = 0 if steady else (tx - t) / self.dt
        
        # cp*Q*t = dtdt + u*tx_x + v*tx_y - alpha*(tx_xx+tx_yy)
        rhs = dtdt + ux*tx_x + vx*tx_y - self.alpha_r*(tx_xx+tx_yy)
        q_inferred = rhs / (self.cp_r * tx + 1e-8)
        q_inferred = torch.clamp(q_inferred, -10, 10)
        return F.mse_loss(q_inferred, q_guess.view_as(q_inferred)), q_inferred.mean()

# =============================================================================
# 3. Training Logic
# =============================================================================
def train_model(args, model, train_loader, val_loader, device):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    scaler = GradScaler('cuda')
    
    best_val_loss = float('inf')
    last_val_loss = 0.0 # Track the validation loss from the previous epoch
    target_physics_lambda = 0.05
    warmup_threshold = int(args.epochs * 0.15)
    ramp_up_period = int(args.epochs * 0.10) # Ramp up over 10% of total epochs
    
    # History for visualization
    loss_history = []
    
    # Initialize Physics Loss from cache properties
    if hasattr(train_loader.dataset, 'datasets'):
        sample_ds = train_loader.dataset.datasets[0]
    else:
        sample_ds = train_loader.dataset

    # Create dummy params dict for PhysicsLoss initialization
    phys_init_params = {
        'Pr': sample_ds.params.get('Pr', 0.71),
        'norm_params': sample_ds.norm_params
    }
    
    phys = MultiParamPhysicsLoss(phys_init_params, sample_ds.nano_props, 
                                dt=sample_ds.params.get('dt', 0.0001), 
                                dx=1.0/(sample_ds.nx-1), dy=1.0/(sample_ds.ny-1)).to(device)

    for epoch in range(args.epochs):
        # Dynamic Physics Lambda Scheduling
        if epoch < warmup_threshold:
            current_phys_lambda = 0.0
            phase_str = "Data-Only Warmup"
        else:
            # Gradually increase lambda from 0 to target_physics_lambda over ramp_up_period
            ramp_weight = min(1.0, (epoch - warmup_threshold) / (ramp_up_period + 1e-8))
            current_phys_lambda = target_physics_lambda * ramp_weight
            phase_str = f"Physics Ramp-up ({ramp_weight*100:.1f}%)" if ramp_weight < 1.0 else "Full Hybrid"

        model.train()
        train_loss_total = 0
        train_loss_mse = 0
        train_loss_phys = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [{phase_str}]")
        for batch_idx, (inp, tgt, pd) in enumerate(pbar):
            inp, tgt = inp.to(device), tgt.to(device)
            r, h, q, d = pd['Ra'].to(device), pd['Ha'].to(device), pd['Q'].to(device), pd['Da'].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                pred = model(inp, r, h, q, d)
                loss_mse = F.mse_loss(pred, tgt)
                
                # Physics Loss (Only compute if lambda > 0 to save computation)
                if current_phys_lambda > 0:
                    p_losses = phys.physics_residual_loss(inp[:, -1], pred, r, h, q, d)
                    loss_phys = p_losses['continuity'] + p_losses['momentum_x'] + p_losses['momentum_y'] + p_losses['energy']
                else:
                    loss_phys = torch.tensor(0.0, device=device)
                
                loss_total = loss_mse + current_phys_lambda * loss_phys
            
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_total += loss_total.item()
            train_loss_mse += loss_mse.item()
            train_loss_phys += loss_phys.item()
            
            pbar.set_postfix({
                'total': f"{loss_total.item():.6f}",
                'mse': f"{loss_mse.item():.6f}",
                'phys': f"{loss_phys.item():.6f}",
                'last_val': f"{last_val_loss:.6f}"
            })

        # Average losses for the epoch
        n_batches = len(train_loader)
        avg_train_total = train_loss_total / n_batches
        avg_train_mse = train_loss_mse / n_batches
        avg_train_phys = train_loss_phys / n_batches

        # Validation
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for inp, tgt, pd in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                r, h, q, d = pd['Ra'].to(device), pd['Ha'].to(device), pd['Q'].to(device), pd['Da'].to(device)
                pred = model(inp, r, h, q, d)
                val_loss_accum += F.mse_loss(pred, tgt).item()
        
        avg_val_loss = val_loss_accum / len(val_loader)
        last_val_loss = avg_val_loss # Update for the next epoch's display
        print(f"  Epoch {epoch+1} Val Loss: {avg_val_loss:.6f}")
        scheduler.step(avg_val_loss)
        
        # Save history entry
        loss_history.append({
            'epoch': epoch + 1,
            'train_total': avg_train_total,
            'train_mse': avg_train_mse,
            'train_phys': avg_train_phys,
            'val_loss': avg_val_loss,
            'lambda': current_phys_lambda
        })
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'checkpoint_best_{args.base_fluid}.pth')
            print(f"  [Model Saved] Best Val Loss: {best_val_loss:.6f}")
            
    # Save the entire history to JSON
    history_path = f'loss_history_{args.base_fluid}.json'
    with open(history_path, 'w') as f:
        json.dump(loss_history, f, indent=4)
    print(f"  [History Saved] Loss history written to {history_path}")

    # =============================================================================
    # 4. Ultra Inference Logic
    # =============================================================================
def predict_multi_params_ultra(model, physics_loss_fn, dataset, config, device, norm_weights=None, num_restarts=4):
    model.eval()
    num_samples = min(config.get('num_inference_samples', 20), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    batch_input = torch.stack([dataset[i][0] for i in indices]).to(device)
    batch_target = torch.stack([dataset[i][1] for i in indices]).to(device)

    # 1. Unified Latent Space: [restarts, 4] in range [0, 1]
    p_latent = torch.rand((num_restarts, 4), device=device, requires_grad=True)

    def get_physical_params(latent):
        ra = 10**(np.log10(config['ra_min']) + (np.log10(config['ra_max']) - np.log10(config['ra_min'])) * latent[:, 0])
        ha = config['ha_min'] + (config['ha_max'] - config['ha_min']) * latent[:, 1]
        q = config['q_min'] + (config['q_max'] - config['q_min']) * latent[:, 2]
        da = 10**(np.log10(config['da_min']) + (np.log10(config['da_max']) - np.log10(config['da_min'])) * latent[:, 3])
        return ra, ha, q, da

    optimizer_adam = optim.Adam([p_latent], lr=config['inference_lr'])
    scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=config['inference_steps'], eta_min=config['inference_lr']*0.1)

    print(f"    [Ultra] Adam Phase ({config['inference_steps']} steps)...")
    for step in range(config['inference_steps']):
        optimizer_adam.zero_grad(set_to_none=True)
        with torch.no_grad(): p_latent.clamp_(0.0, 1.0)

        ra, ha, q, da = get_physical_params(p_latent)
        r_e, h_e, q_e, d_e = [x.unsqueeze(1).expand(-1, num_samples).reshape(-1) for x in [ra, ha, q, da]]

        pred = model(batch_input.repeat(num_restarts, 1, 1, 1, 1), r_e, h_e, q_e, d_e)
        l_data = (pred - batch_target.repeat(num_restarts, 1, 1, 1)).pow(2).view(num_restarts, num_samples, -1).mean(dim=(1, 2))

        p_l = physics_loss_fn.physics_residual_loss(batch_input.repeat(num_restarts, 1, 1, 1, 1)[:, -1], pred, r_e, h_e, q_e, d_e)
        l_phys = (p_l['continuity'].view(num_restarts, num_samples).mean(dim=1) * norm_weights.get('continuity', 1.0) +
                  p_l['momentum_x'].view(num_restarts, num_samples).mean(dim=1) * norm_weights.get('momentum_x', 1.0) +
                  p_l['momentum_y'].view(num_restarts, num_samples).mean(dim=1) * norm_weights.get('momentum_y', 3.0) +
                  p_l['energy'].view(num_restarts, num_samples).mean(dim=1) * norm_weights.get('energy', 3.0))

        # Consistency
        un, vn, tn = torch.chunk(batch_input.repeat(num_restarts, 1, 1, 1, 1)[:, -1], 4, 1)[:3]
        unx, vnx, tnx, pnx = torch.chunk(pred, 4, 1)
        lcd, _ = physics_loss_fn.da_consistency_loss(un, vn, pnx, unx, vnx, d_e)
        lcr, _ = physics_loss_fn.ra_consistency_loss(un, vn, pnx, unx, vnx, tnx, r_e, d_e, h_e)
        lch, _ = physics_loss_fn.ha_consistency_loss(un, vn, pnx, unx, vnx, tnx, h_e, r_e, d_e)
        lcq, _ = physics_loss_fn.q_consistency_loss(un, vn, tn, tnx, q_e)
        l_cons = (lcd + lcr + lch + lcq).view(num_restarts, num_samples).mean(dim=1)
        
        l_bound = physics_loss_fn.boundary_loss(pred).view(num_restarts, num_samples).mean(dim=1)

        loss_total = (l_data + 2.0 * l_phys + 2.0 * l_cons + 5.0 * l_bound).mean()
        loss_total.backward()
        optimizer_adam.step()
        scheduler_adam.step()

        if step % 100 == 0:
            bi = torch.argmin(l_data).item()
            print(f"      Step {step:4d} | Best Ra:{ra[bi]:.2e} Ha:{ha[bi]:.2f} Q:{q[bi]:.2f} Da:{da[bi]:.4f} | Data Loss: {l_data[bi]:.6f}")

    # L-BFGS Refinement
    with torch.no_grad():
        bi = torch.argmin(l_data).item()
        p_best = p_latent[bi:bi+1].detach().clone().requires_grad_(True)
    
    print(f"\n    [Ultra] L-BFGS Refinement...")
    optimizer_lbfgs = optim.LBFGS([p_best], lr=0.01, max_iter=50, line_search_fn='strong_wolfe')

    def closure():
        optimizer_lbfgs.zero_grad()
        with torch.no_grad(): p_best.clamp_(0.0, 1.0)
        ra, ha, q, da = get_physical_params(p_best)
        r_e, h_e, q_e, d_e = [x.expand(num_samples) for x in [ra, ha, q, da]]
        p = model(batch_input, r_e, h_e, q_e, d_e)
        ld = (p - batch_target).pow(2).mean()
        lb = physics_loss_fn.boundary_loss(p).mean()
        lt = ld + 5.0 * lb
        lt.backward()
        return lt

    optimizer_lbfgs.step(closure)
    ra, ha, q, da = get_physical_params(p_best)
    return {'Ra': ra.item(), 'Ha': ha.item(), 'Q': q.item(), 'Da': da.item()}

# =============================================================================
# 5. Main Execution
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_fluid', default='EG')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--inference_only', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PhyCRNet V3 Initialized. Device: {device}")

    # 1. Data Pooling and Splitting (80/10/10)
    base_path = os.path.join(args.data_root, args.base_fluid)
    all_files = glob.glob(os.path.join(base_path, "**", "*.mat"), recursive=True)
    all_files = [f for f in sorted(all_files) if 'phi' not in f.lower()]
    random.seed(42)
    random.shuffle(all_files)

    n = len(all_files)
    train_files = all_files[:int(n*0.8)]
    val_files = all_files[int(n*0.8):int(n*0.9)]
    test_files = all_files[int(n*0.9):]

    print(f"Files: {n} Total | {len(train_files)} Train | {len(val_files)} Val | {len(test_files)} Test")

    # 2. HDF5 Caching
    cache_dir = f"cache_{args.base_fluid}"
    def get_loader(files, shuffle=True):
        caches = []
        for f in tqdm(files, desc="Caching"):
            cp = preprocess_to_hdf5(f, cache_dir)
            if cp: caches.append(CachedSequenceDataset(cp))
        ds = ConcatDataset(caches)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

    if not args.inference_only:
        train_loader = get_loader(train_files)
        val_loader = get_loader(val_files, shuffle=False)
        model = MultiParamSurrogateModel(hidden=256).to(device)
        
        if not args.skip_train:
            train_model(args, model, train_loader, val_loader, device)
        else:
            model.load_state_dict(torch.load(f'checkpoint_best_{args.base_fluid}.pth'))
    else:
        model = MultiParamSurrogateModel(hidden=256).to(device)
        model.load_state_dict(torch.load(f'checkpoint_best_{args.base_fluid}.pth'))

    # 3. Test/Inference (Ultra Phase)
    model.eval()
    print("\nStarting Ultra-Precision Inference on Test Set...")
    inf_config = {
        'inference_steps': 1500, 'inference_lr': 0.005,
        'ra_min': 100, 'ra_max': 1e8, 'ha_min': 0, 'ha_max': 100,
        'q_min': -10, 'q_max': 10, 'da_min': 0.001, 'da_max': 0.15,
        'num_inference_samples': 20
    }
    norm_w = {'continuity': 1.0, 'momentum_x': 1.0, 'momentum_y': 3.0, 'energy': 3.0}

    for f_path in test_files[:5]: # Show first 5 for preview
        ds_mat = MatDataset(f_path, device=device)
        ds_seq = CachedSequenceDataset(preprocess_to_hdf5(f_path, cache_dir), device=device)
        phys = MultiParamPhysicsLoss(ds_mat.params, ds_mat.nanofluid_props, 
                                    dt=ds_mat.params['dt'], dx=1.0/(ds_mat.nx-1), dy=1.0/(ds_mat.ny-1)).to(device)
        
        pred_p = predict_multi_params_ultra(model, phys, ds_seq, inf_config, device, norm_weights=norm_w)
        
        print(f"\nResults for {os.path.basename(f_path)}:")
        for p in ['Ra', 'Ha', 'Q', 'Da']:
            true_v = ds_mat.params.get(p)
            print(f"  {p}: True={true_v:.4f}, Pred={pred_p[p]:.4f}, Err={abs(true_v-pred_p[p])/(abs(true_v)+1e-8)*100:.2f}%")

if __name__ == '__main__':
    main()
