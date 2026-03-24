"""
PhyCRNet Multi-Parameter Solver V3.1 (RTX 4090 Optimized & Bug Fixed)
=====================================================================
High-performance training and inference for fluid parameters (Ra, Ha, Q, Da).

Key Upgrades & Fixes:
- [FIXED] Batch dimension preservation in MultiParamPhysicsLoss for Ultra Inference
- [FIXED] Leaf tensor in-place modification bug in Adam/L-BFGS loops
- [FIXED] FP16 instability bypassed by casting physics calculations to FP32
- [FIXED] Unnorm dummy variable typos corrected for strict mathematical consistency
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
            'Q':  torch.tensor(self.params.get('Q', 0.0), dtype=torch.float32),
            'Da': torch.tensor(self.params.get('Da', 1e-3), dtype=torch.float32)
        }
        return inp, tgt, pd

# =============================================================================
# 2. Physics Loss Components (Fixed Batch Dimension Handling)
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
        """Penalty for non-zero velocity at walls. Preserves Batch [B] dimension."""
        u, v, t, p = torch.chunk(field, 4, 1)
        # mean(dim=[1,2]) reduces C, H or W, leaving B
        loss_u = (u[:, :, 0, :].pow(2).mean(dim=[1, 2]) + u[:, :, -1, :].pow(2).mean(dim=[1, 2]) + 
                  u[:, :, :, 0].pow(2).mean(dim=[1, 2]) + u[:, :, :, -1].pow(2).mean(dim=[1, 2]))
        loss_v = (v[:, :, 0, :].pow(2).mean(dim=[1, 2]) + v[:, :, -1, :].pow(2).mean(dim=[1, 2]) + 
                  v[:, :, :, 0].pow(2).mean(dim=[1, 2]) + v[:, :, :, -1].pow(2).mean(dim=[1, 2]))
        return loss_u + loss_v

    def physics_residual_loss(self, inp_t, pred, r, h, q, d, steady=False):
        """Returns Physics Loss while preserving Batch [B] dimension."""
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

        rb = d.view(-1, 1, 1, 1); rab = r.view(-1, 1, 1, 1)
        hab = h.view(-1, 1, 1, 1); qb = q.view(-1, 1, 1, 1)

        res_x = (dudt + u_x*ux_x + v_x*ux_y) - (-px_x + self.nu_r*self.Pr*(ux_xx+ux_yy) - (self.nu_r*self.Pr/rb)*u_x)
        res_y = (dvdt + u_x*vx_x + v_x*vx_y) - (-px_y + self.nu_r*self.Pr*(vx_xx+vx_yy) + self.beta_r*rab*self.Pr*t_x - (self.nu_r*self.Pr/rb)*v_x - (self.sigma_r*self.rho_r*hab**2*self.Pr)*v_x)
        res_e = (dtdt + u_x*tx_x + v_x*tx_y) - (self.alpha_r*(tx_xx+tx_yy) + self.cp_r*qb*t_x)

        # mean(dim=[1, 2, 3]) leaves [B]
        return {
            'continuity': res_c.pow(2).mean(dim=[1, 2, 3]),
            'momentum_x': res_x.pow(2).mean(dim=[1, 2, 3]),
            'momentum_y': res_y.pow(2).mean(dim=[1, 2, 3]),
            'energy':     res_e.pow(2).mean(dim=[1, 2, 3])
        }

    # Consistency Loss Utilities (Stabilized with Log-scale and Normalization)
    def da_consistency_loss(self, un, vn, pn_x, un_x, vn_x, d_guess, steady=False):
        z_t = torch.zeros_like(un)
        u, v, _, _ = self.unnorm(un, vn, z_t, z_t)
        ux, vx, _, px = self.unnorm(un_x, vn_x, z_t, pn_x)
        ux_x, ux_y, ux_xx, ux_yy = self.compute_derivatives(ux)
        px_x, _, _, _ = self.compute_derivatives(px)
        dudt = 0 if steady else (ux - u) / self.dt
        
        rhs = dudt + ux*ux_x + vx*ux_y + px_x - self.nu_r*self.Pr*(ux_xx+ux_yy)
        # Numerical stability: avoid division by zero and extreme values
        da_inferred = -(self.nu_r * self.Pr * ux) / (rhs + 1e-7)
        da_inferred = torch.clamp(da_inferred, 1e-5, 1.0)

        # Log-scale comparison for Da (0.001 to 0.15)
        log_da_inf = torch.log10(da_inferred)
        log_da_tgt = torch.log10(torch.clamp(d_guess.view(-1, 1, 1, 1), 1e-5, 1.0)).expand_as(log_da_inf)
        mse = F.mse_loss(log_da_inf, log_da_tgt, reduction='none').mean(dim=[1, 2, 3])
        return mse, da_inferred.mean(dim=[1, 2, 3])

    def ra_consistency_loss(self, un, vn, pn_x, un_x, vn_x, tn_x, r_guess, d_val, h_val, steady=False):
        z_t = torch.zeros_like(un)
        u, v, _, _ = self.unnorm(un, vn, z_t, z_t)
        ux, vx, tx, px = self.unnorm(un_x, vn_x, tn_x, pn_x)
        vx_x, vx_y, vx_xx, vx_yy = self.compute_derivatives(vx)
        px_y, _, _, _ = self.compute_derivatives(px)
        dvdt = 0 if steady else (vx - v) / self.dt
        
        rb = d_val.view(-1, 1, 1, 1); hab = h_val.view(-1, 1, 1, 1)
        rhs = dvdt + ux*vx_x + vx*vx_y + px_y - self.nu_r*self.Pr*(vx_xx+vx_yy) + (self.nu_r*self.Pr/rb)*vx + (self.sigma_r*self.rho_r*hab**2*self.Pr)*vx
        ra_inferred = rhs / (self.beta_r * self.Pr * tx + 1e-7)
        ra_inferred = torch.clamp(ra_inferred, 10.0, 1e9)
        
        # Log-scale comparison for Ra (1e2 to 1e8)
        log_ra_inf = torch.log10(ra_inferred)
        log_ra_tgt = torch.log10(r_guess.view(-1, 1, 1, 1)).expand_as(log_ra_inf)
        mse = F.mse_loss(log_ra_inf, log_ra_tgt, reduction='none').mean(dim=[1, 2, 3])
        return mse, ra_inferred.mean(dim=[1, 2, 3])

    def ha_consistency_loss(self, un, vn, pn_x, un_x, vn_x, tn_x, h_guess, r_val, d_val, steady=False):
        z_t = torch.zeros_like(un)
        u, v, _, _ = self.unnorm(un, vn, z_t, z_t)
        ux, vx, tx, px = self.unnorm(un_x, vn_x, tn_x, pn_x)
        vx_x, vx_y, vx_xx, vx_yy = self.compute_derivatives(vx)
        px_y, _, _, _ = self.compute_derivatives(px)
        dvdt = 0 if steady else (vx - v) / self.dt
        
        rb = d_val.view(-1, 1, 1, 1); rab = r_val.view(-1, 1, 1, 1)
        rhs = -(dvdt + ux*vx_x + vx*vx_y + px_y - self.nu_r*self.Pr*(vx_xx+vx_yy) - self.beta_r*rab*self.Pr*tx + (self.nu_r*self.Pr/rb)*vx)
        ha_sq_inferred = rhs / (self.sigma_r * self.rho_r * self.Pr * vx + 1e-7)
        ha_inferred = torch.sqrt(torch.clamp(ha_sq_inferred, 0, 1e5))
        
        # Normalized comparison for Ha (0 to 100)
        h_target = h_guess.view(-1, 1, 1, 1).expand_as(ha_inferred)
        mse = F.mse_loss(ha_inferred / 100.0, h_target / 100.0, reduction='none').mean(dim=[1, 2, 3])
        return mse, ha_inferred.mean(dim=[1, 2, 3])

    def q_consistency_loss(self, un, vn, tn, tn_x, q_guess, steady=False):
        z_t = torch.zeros_like(un)
        u, v, t, _ = self.unnorm(un, vn, tn, z_t)
        ux, vx, tx, _ = self.unnorm(un, vn, tn_x, z_t)
        tx_x, tx_y, tx_xx, tx_yy = self.compute_derivatives(tx)
        dtdt = 0 if steady else (tx - t) / self.dt
        
        rhs = dtdt + ux*tx_x + vx*tx_y - self.alpha_r*(tx_xx+tx_yy)
        q_inferred = rhs / (self.cp_r * tx + 1e-7)
        q_inferred = torch.clamp(q_inferred, -20, 20)
        
        # Normalized comparison for Q (-10 to 10)
        q_target = q_guess.view(-1, 1, 1, 1).expand_as(q_inferred)
        mse = F.mse_loss(q_inferred / 10.0, q_target / 10.0, reduction='none').mean(dim=[1, 2, 3])
        return mse, q_inferred.mean(dim=[1, 2, 3])

# =============================================================================
# 3. Training Logic (Fixed FP16 Instability & Auto-Normalization)
# =============================================================================
def calculate_physics_normalization(model, dataloader, phys_fn, device, num_batches=3):
    """Measures initial physics loss scales to prevent exploding gradients."""
    print(f"\n  [Setup] Calculating Physics Normalization Weights ({num_batches} batches)...")
    model.eval()
    accum = {'continuity': [], 'momentum_x': [], 'momentum_y': [], 'energy': []}
    
    with torch.no_grad():
        for i, (inp, tgt, pd) in enumerate(dataloader):
            if i >= num_batches: break
            inp = inp.to(device)
            r, h, q, d = pd['Ra'].to(device), pd['Ha'].to(device), pd['Q'].to(device), pd['Da'].to(device)
            
            with autocast('cuda'):
                pred = model(inp, r, h, q, d)
            
            p_losses = phys_fn.physics_residual_loss(inp[:, -1].float(), pred.float(), r, h, q, d)
            for k in accum.keys():
                val = p_losses[k].mean().item()
                if np.isfinite(val) and val > 0:
                    accum[k].append(val)
    
    weights = {}
    for k, vals in accum.items():
        mean_val = np.mean(vals) if vals else 1.0
        # Target scale is ~0.1 to match initial MSE roughly
        weights[k] = 1.0 / (mean_val + 1e-9)
        print(f"    - {k:10s} | Raw Mean: {mean_val:.2e} | Norm Weight: {weights[k]:.2e}")
    
    return weights

def train_model(args, model, train_loader, val_loader, device):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    scaler = GradScaler('cuda')
    
    # 1. Physics Normalization
    if hasattr(train_loader.dataset, 'datasets'):
        sample_ds = train_loader.dataset.datasets[0]
    else:
        sample_ds = train_loader.dataset

    phys_init_params = {
        'Pr': sample_ds.params.get('Pr', 0.71),
        'norm_params': sample_ds.norm_params
    }
    
    phys = MultiParamPhysicsLoss(phys_init_params, sample_ds.nano_props, 
                                dt=sample_ds.params.get('dt', 0.0001), 
                                dx=1.0/(sample_ds.nx-1), dy=1.0/(sample_ds.ny-1)).to(device)

    norm_weights = calculate_physics_normalization(model, train_loader, phys, device)

    # 2. Pre-train Check
    print("\n  [Check] Pre-train Physics Loss Verification:")
    model.eval()
    with torch.no_grad():
        inp, tgt, pd = next(iter(train_loader))
        inp = inp.to(device); r, h, q, d = pd['Ra'].to(device), pd['Ha'].to(device), pd['Q'].to(device), pd['Da'].to(device)
        pred = model(inp, r, h, q, d)
        p_l = phys.physics_residual_loss(inp[:, -1].float(), pred.float(), r, h, q, d)
        
        raw_total = sum(v.mean().item() for v in p_l.values())
        norm_total = sum(p_l[k].mean().item() * norm_weights[k] for k in p_l.keys())
        print(f"    - Raw Total Physics Loss:  {raw_total:.2e}")
        print(f"    - Norm Total Physics Loss: {norm_total:.6f} (Should be near 1.0~4.0)")
        print(f"    - Initial Data Loss (MSE): {F.mse_loss(pred, tgt.to(device)).item():.6f}")

    best_val_loss = float('inf')
    last_val_loss = 0.0 
    target_physics_lambda = 0.05
    warmup_threshold = int(args.epochs * 0.15)
    ramp_up_period = int(args.epochs * 0.10)
    
    loss_history = []

    for epoch in range(args.epochs):
        if epoch < warmup_threshold:
            current_phys_lambda = 0.0
            phase_str = "Data-Only Warmup"
        else:
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
                
            if current_phys_lambda > 0:
                p_losses = phys.physics_residual_loss(inp[:, -1].float(), pred.float(), r, h, q, d)
                # Apply Normalization Weights
                loss_phys = (p_losses['continuity'].mean() * norm_weights['continuity'] + 
                            p_losses['momentum_x'].mean() * norm_weights['momentum_x'] + 
                            p_losses['momentum_y'].mean() * norm_weights['momentum_y'] + 
                            p_losses['energy'].mean() * norm_weights['energy'])
            else:
                loss_phys = torch.tensor(0.0, device=device)
                
            loss_total = loss_mse + current_phys_lambda * loss_phys
            
            # 3. Backward
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
        last_val_loss = avg_val_loss
        print(f"  Epoch {epoch+1} Val Loss: {avg_val_loss:.6f}")
        scheduler.step(avg_val_loss)
        
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
            
    history_path = f'loss_history_{args.base_fluid}.json'
    with open(history_path, 'w') as f:
        json.dump(loss_history, f, indent=4)
    print(f"  [History Saved] Loss history written to {history_path}")

# =============================================================================
# 4. Ultra Inference Logic (Fixed Leaf Tensor In-Place Mod & Reshaping)
# =============================================================================
def predict_multi_params_ultra(model, physics_loss_fn, dataset, config, device, norm_weights=None, num_restarts=4, log_prefix="inf"):
    model.eval()
    num_samples = min(config.get('num_inference_samples', 20), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    batch_input = torch.stack([dataset[i][0] for i in indices]).to(device)
    batch_target = torch.stack([dataset[i][1] for i in indices]).to(device)

    # 0. Initialize Log File
    os.makedirs("inference_logs", exist_ok=True)
    log_path = os.path.join("inference_logs", f"{log_prefix}_steps.txt")
    log_f = open(log_path, "w")
    def log_print(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log_print(f"    [Ultra] Starting Inference for prefix: {log_prefix}")

    # 1. Automatic Physics Normalization for Inference
    log_print("    [Ultra] Calculating Case-Specific Physics Normalization...")
    with torch.no_grad():
        # Use initial p_raw to estimate scales
        p_raw_sample = torch.randn((num_restarts, 4), device=device)
        p_lat_sample = torch.sigmoid(p_raw_sample)
        def get_params_local(latent): # Local helper for scaling
            ra = 10**(np.log10(config['ra_min']) + (np.log10(config['ra_max']) - np.log10(config['ra_min'])) * latent[:, 0])
            ha = config['ha_min'] + (config['ha_max'] - config['ha_min']) * latent[:, 1]
            q = config['q_min'] + (config['q_max'] - config['q_min']) * latent[:, 2]
            da = 10**(np.log10(config['da_min']) + (np.log10(config['da_max']) - np.log10(config['da_min'])) * latent[:, 3])
            return ra, ha, q, da
        ra_s, ha_s, q_s, da_s = get_params_local(p_lat_sample)
        r_es, h_es, q_es, d_es = [x.unsqueeze(1).expand(-1, num_samples).reshape(-1) for x in [ra_s, ha_s, q_s, da_s]]
        pred_s = model(batch_input.repeat(num_restarts, 1, 1, 1, 1), r_es, h_es, q_es, d_es)
        p_l_s = physics_loss_fn.physics_residual_loss(batch_input.repeat(num_restarts, 1, 1, 1, 1)[:, -1], pred_s, r_es, h_es, q_es, d_es)
        
        # Calculate normalization weights: target scale is ~0.1
        actual_norm_weights = {}
        for k in ['continuity', 'momentum_x', 'momentum_y', 'energy']:
            raw_val = p_l_s[k].mean().item()
            # If norm_weights was passed, treat it as relative importance (e.g. 3.0 for energy)
            rel_importance = norm_weights.get(k, 1.0) if norm_weights else 1.0
            actual_norm_weights[k] = rel_importance / (raw_val + 1e-9)
            log_print(f"      - {k:10s} | Raw Scale: {raw_val:.2e} | Final Weight: {actual_norm_weights[k]:.2e}")

    # Use the calculated weights for the rest of the function
    norm_weights = actual_norm_weights

    # 2. Unified Latent Space: [restarts, 4] transformed via sigmoid to [0, 1]
    p_raw = torch.randn((num_restarts, 4), device=device, requires_grad=True)

    def get_physical_params(latent):
        ra = 10**(np.log10(config['ra_min']) + (np.log10(config['ra_max']) - np.log10(config['ra_min'])) * latent[:, 0])
        ha = config['ha_min'] + (config['ha_max'] - config['ha_min']) * latent[:, 1]
        q = config['q_min'] + (config['q_max'] - config['q_min']) * latent[:, 2]
        da = 10**(np.log10(config['da_min']) + (np.log10(config['da_max']) - np.log10(config['da_min'])) * latent[:, 3])
        return ra, ha, q, da

    optimizer_adam = optim.Adam([p_raw], lr=config['inference_lr'])
    scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=config['inference_steps'], eta_min=config['inference_lr']*0.1)

    # Initial loss calculation to ensure l_data is always defined
    with torch.no_grad():
        p_latent_init = torch.sigmoid(p_raw)
        ra_i, ha_i, q_i, da_i = get_physical_params(p_latent_init)
        r_ei, h_ei, q_ei, d_ei = [x.unsqueeze(1).expand(-1, num_samples).reshape(-1) for x in [ra_i, ha_i, q_i, da_i]]
        pred_i = model(batch_input.repeat(num_restarts, 1, 1, 1, 1), r_ei, h_ei, q_ei, d_ei)
        l_data = (pred_i - batch_target.repeat(num_restarts, 1, 1, 1)).pow(2).view(num_restarts, num_samples, -1).mean(dim=(1, 2))

    log_print(f"    [Ultra] Adam Phase ({config['inference_steps']} steps)...")
    warmup_steps = 500 # Steps to focus only on Data scale
    for step in range(config['inference_steps']):
        optimizer_adam.zero_grad(set_to_none=True)
        
        p_latent = torch.sigmoid(p_raw)
        ra, ha, q, da = get_physical_params(p_latent)
        r_e, h_e, q_e, d_e = [x.unsqueeze(1).expand(-1, num_samples).reshape(-1) for x in [ra, ha, q, da]]

        pred = model(batch_input.repeat(num_restarts, 1, 1, 1, 1), r_e, h_e, q_e, d_e)
        l_data = (pred - batch_target.repeat(num_restarts, 1, 1, 1)).pow(2).view(num_restarts, num_samples, -1).mean(dim=(1, 2))

        # Phase selection
        if step < warmup_steps:
            # Stage 1: Focus purely on matching the data pattern to get Ra scale right
            loss_total = l_data.mean()
            l_phys = torch.zeros_like(l_data)
            l_cons = torch.zeros_like(l_data)
            l_bound = torch.zeros_like(l_data)
            # Dummy values for logging
            inf_ra = ra; inf_ha = ha; inf_q = q; inf_da = da
            lcr_r = l_cons; lch_r = l_cons; lcq_r = l_cons; lcd_r = l_cons
        else:
            # Stage 2: Refine with Physics and Consistency
            p_l = physics_loss_fn.physics_residual_loss(batch_input.repeat(num_restarts, 1, 1, 1, 1)[:, -1], pred, r_e, h_e, q_e, d_e)
            l_phys = (p_l['continuity'].view(num_restarts, num_samples).mean(dim=1) * norm_weights.get('continuity', 1.0) +
                      p_l['momentum_x'].view(num_restarts, num_samples).mean(dim=1) * norm_weights.get('momentum_x', 1.0) +
                      p_l['momentum_y'].view(num_restarts, num_samples).mean(dim=1) * norm_weights.get('momentum_y', 3.0) +
                      p_l['energy'].view(num_restarts, num_samples).mean(dim=1) * norm_weights.get('energy', 3.0))

            un, vn, tn = torch.chunk(batch_input.repeat(num_restarts, 1, 1, 1, 1)[:, -1], 4, 1)[:3]
            unx, vnx, tnx, pnx = torch.chunk(pred, 4, 1)
            lcd, inf_da = physics_loss_fn.da_consistency_loss(un, vn, pnx, unx, vnx, d_e)
            lcr, inf_ra = physics_loss_fn.ra_consistency_loss(un, vn, pnx, unx, vnx, tnx, r_e, d_e, h_e)
            lch, inf_ha = physics_loss_fn.ha_consistency_loss(un, vn, pnx, unx, vnx, tnx, h_e, r_e, d_e)
            lcq, inf_q = physics_loss_fn.q_consistency_loss(un, vn, tn, tnx, q_e)
            
            lcd_r = lcd.view(num_restarts, num_samples).mean(dim=1)
            lcr_r = lcr.view(num_restarts, num_samples).mean(dim=1)
            lch_r = lch.view(num_restarts, num_samples).mean(dim=1)
            lcq_r = lcq.view(num_restarts, num_samples).mean(dim=1)
            l_cons = lcd_r + lcr_r + lch_r + lcq_r
            l_bound = physics_loss_fn.boundary_loss(pred).view(num_restarts, num_samples).mean(dim=1)

            # High Data weight (50.0) and lower Physics/Cons to prevent Ra drop
            loss_total = (50.0 * l_data + 0.5 * l_phys + 0.1 * l_cons + 2.0 * l_bound).mean()

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_([p_raw], max_norm=1.0)
        optimizer_adam.step()
        scheduler_adam.step()

        if step % 10 == 0:
            with torch.no_grad():
                # During warmup, pick best based on Data. After, pick based on Weighted Total.
                sel_metric = l_data if step < warmup_steps else (20.0 * l_data + 1.0 * l_phys)
                bi = torch.argmin(sel_metric).item()
            
            phase = "WARMUP" if step < warmup_steps else "HYBRID"
            log_print(f"      Step {step:4d} [{phase}] | Guess: Ra:{ra[bi].item():.2e} Ha:{ha[bi].item():.2f} Q:{q[bi].item():.2f} Da:{da[bi].item():.4f}")
            if step >= warmup_steps:
                log_print(f"             | Infer: Ra:{inf_ra[bi].item():.2e} Ha:{inf_ha[bi].item():.2f} Q:{inf_q[bi].item():.2f} Da:{inf_da[bi].item():.4f}")
                log_print(f"             | C-Loss:Ra:{lcr_r[bi].item():.4f} Ha:{lch_r[bi].item():.4f} Q:{lcq_r[bi].item():.4f} Da:{lcd_r[bi].item():.4f}")
            log_print(f"             | LOSSES: Data:{l_data[bi].item():.4f} Phys:{l_phys[bi].item():.4f} Total:{loss_total.item():.4f}")

    # L-BFGS Refinement
    with torch.no_grad():
        bi = torch.argmin(l_data).item()
        p_best_raw = p_raw[bi:bi+1].detach().clone().requires_grad_(True)
    
    lbfgs_iters = config.get('lbfgs_steps', 100)
    log_print(f"\n    [Ultra] L-BFGS Refinement (Max {lbfgs_iters} iters)...")
    optimizer_lbfgs = optim.LBFGS([p_best_raw], lr=1.0, max_iter=lbfgs_iters, line_search_fn='strong_wolfe')

    iter_count = 0
    def closure():
        nonlocal iter_count
        optimizer_lbfgs.zero_grad()
        p_latent_best = torch.sigmoid(p_best_raw)
        ra, ha, q, da = get_physical_params(p_latent_best)
        r_e, h_e, q_e, d_e = [x.expand(num_samples) for x in [ra, ha, q, da]]
        
        pred = model(batch_input, r_e, h_e, q_e, d_e)
        
        ld = (pred - batch_target).pow(2).mean()
        
        p_l = physics_loss_fn.physics_residual_loss(batch_input[:, -1], pred, r_e, h_e, q_e, d_e)
        l_phys = (p_l['continuity'].mean() * norm_weights.get('continuity', 1.0) +
                  p_l['momentum_x'].mean() * norm_weights.get('momentum_x', 1.0) +
                  p_l['momentum_y'].mean() * norm_weights.get('momentum_y', 3.0) +
                  p_l['energy'].mean() * norm_weights.get('energy', 3.0))

        un, vn, tn = torch.chunk(batch_input[:, -1], 4, 1)[:3]
        unx, vnx, tnx, pnx = torch.chunk(pred, 4, 1)
        lcd, _ = physics_loss_fn.da_consistency_loss(un, vn, pnx, unx, vnx, d_e)
        lcr, _ = physics_loss_fn.ra_consistency_loss(un, vn, pnx, unx, vnx, tnx, r_e, d_e, h_e)
        lch, _ = physics_loss_fn.ha_consistency_loss(un, vn, pnx, unx, vnx, tnx, h_e, r_e, d_e)
        lcq, _ = physics_loss_fn.q_consistency_loss(un, vn, tn, tnx, q_e)
        l_cons = (lcd + lcr + lch + lcq).mean()
        
        lb = physics_loss_fn.boundary_loss(pred).mean()
        
        lt = ld + 2.0 * l_phys + 2.0 * l_cons + 5.0 * lb
        lt.backward()

        if iter_count % 10 == 0:
            log_print(f"      Iter {iter_count:3d} | Ra:{ra.item():.2e} Ha:{ha.item():.2f} Q:{q.item():.2f} Da:{da.item():.4f} | Loss: {lt.item():.6f}")
        
        iter_count += 1
        return lt

    optimizer_lbfgs.step(closure)
    p_latent_final = torch.sigmoid(p_best_raw)
    ra, ha, q, da = get_physical_params(p_latent_final)
    
    log_f.close()
    return {'Ra': ra.item(), 'Ha': ha.item(), 'Q': q.item(), 'Da': da.item()}

    # L-BFGS Refinement
    with torch.no_grad():
        bi = torch.argmin(l_data).item()
        p_best_raw = p_raw[bi:bi+1].detach().clone().requires_grad_(True)
    
    lbfgs_iters = config.get('lbfgs_steps', 100)
    print(f"\n    [Ultra] L-BFGS Refinement (Max {lbfgs_iters} iters)...")
    optimizer_lbfgs = optim.LBFGS([p_best_raw], lr=1.0, max_iter=lbfgs_iters, line_search_fn='strong_wolfe')

    iter_count = 0
    def closure():
        nonlocal iter_count
        optimizer_lbfgs.zero_grad()
        p_latent_best = torch.sigmoid(p_best_raw)
        ra, ha, q, da = get_physical_params(p_latent_best)
        r_e, h_e, q_e, d_e = [x.expand(num_samples) for x in [ra, ha, q, da]]
        
        pred = model(batch_input, r_e, h_e, q_e, d_e)
        
        ld = (pred - batch_target).pow(2).mean()
        
        p_l = physics_loss_fn.physics_residual_loss(batch_input[:, -1], pred, r_e, h_e, q_e, d_e)
        l_phys = (p_l['continuity'].mean() * norm_weights.get('continuity', 1.0) +
                  p_l['momentum_x'].mean() * norm_weights.get('momentum_x', 1.0) +
                  p_l['momentum_y'].mean() * norm_weights.get('momentum_y', 3.0) +
                  p_l['energy'].mean() * norm_weights.get('energy', 3.0))

        un, vn, tn = torch.chunk(batch_input[:, -1], 4, 1)[:3]
        unx, vnx, tnx, pnx = torch.chunk(pred, 4, 1)
        lcd, _ = physics_loss_fn.da_consistency_loss(un, vn, pnx, unx, vnx, d_e)
        lcr, _ = physics_loss_fn.ra_consistency_loss(un, vn, pnx, unx, vnx, tnx, r_e, d_e, h_e)
        lch, _ = physics_loss_fn.ha_consistency_loss(un, vn, pnx, unx, vnx, tnx, h_e, r_e, d_e)
        lcq, _ = physics_loss_fn.q_consistency_loss(un, vn, tn, tnx, q_e)
        l_cons = (lcd + lcr + lch + lcq).mean()
        
        lb = physics_loss_fn.boundary_loss(pred).mean()
        
        lt = ld + 2.0 * l_phys + 2.0 * l_cons + 5.0 * lb
        lt.backward()

        if iter_count % 10 == 0:
            print(f"      Iter {iter_count:3d} | Ra:{ra.item():.2e} Ha:{ha.item():.2f} Q:{q.item():.2f} Da:{da.item():.4f} | Loss: {lt.item():.6f}")
        
        iter_count += 1
        return lt

    optimizer_lbfgs.step(closure)
    p_latent_final = torch.sigmoid(p_best_raw)
    ra, ha, q, da = get_physical_params(p_latent_final)
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
    print(f"PhyCRNet V3.1 Initialized. Device: {device}")

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

    cache_dir = f"cache_{args.base_fluid}"
    def get_loader(files, shuffle=True):
        caches = []
        for f in tqdm(files, desc="Caching"):
            cp = preprocess_to_hdf5(f, cache_dir)
            if cp: caches.append(CachedSequenceDataset(cp))
        ds = ConcatDataset(caches)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)

    if not args.inference_only:
        train_loader = get_loader(train_files)
        val_loader = get_loader(val_files, shuffle=False)
        model = MultiParamSurrogateModel(hidden=256).to(device)
        
        if not args.skip_train:
            train_model(args, model, train_loader, val_loader, device)
        else:
            model.load_state_dict(torch.load(f'checkpoint_best_{args.base_fluid}.pth', weights_only=True))
    else:
        model = MultiParamSurrogateModel(hidden=256).to(device)
        model.load_state_dict(torch.load(f'checkpoint_best_{args.base_fluid}.pth', weights_only=True))

    model.eval()
    print("\nStarting Ultra-Precision Inference on Test Set...")
    inf_config = {
        'inference_steps': 1500, 'inference_lr': 0.005,
        'lbfgs_steps': 100,
        'ra_min': 100, 'ra_max': 1e8, 'ha_min': 0, 'ha_max': 100,
        'q_min': -10, 'q_max': 10, 'da_min': 0.001, 'da_max': 0.15,
        'num_inference_samples': 20
    }
    norm_w = {'continuity': 1.0, 'momentum_x': 1.0, 'momentum_y': 3.0, 'energy': 3.0}

    for f_path in test_files[:5]: 
        ds_mat = MatDataset(f_path, device=device)
        
        # Print Ground Truth first
        print(f"\n" + "="*50)
        print(f"FILE: {os.path.basename(f_path)}")
        print(f"GROUND TRUTH: " + ", ".join([f"{p}={ds_mat.params.get(p):.4f}" for p in ['Ra', 'Ha', 'Q', 'Da']]))
        print("="*50)

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