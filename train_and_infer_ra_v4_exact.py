"""
PhyCRNet Ra-Only EXACT Solver (Based on v4.py)
==============================================
Identical engine to train_and_infer_v4.py.
- Model: MultiParamSurrogateModel (4-params to ensure GPU usage)
- Training: Variable Ra, Fixed GT for Ha, Q, Da
- Inference: Adam + L-BFGS Optimization for Ra ONLY
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import GradScaler, autocast
import numpy as np
import os
import random
import glob
from tqdm import tqdm

from data import MatDataset
from models import MultiParamSurrogateModel
from train_and_infer_v4 import (
    MultiParamPhysicsLoss, preprocess_to_hdf5, CachedSequenceDataset, calculate_physics_normalization
)

# =============================================================================
# 1. Training (Exact same logic as v4.py)
# =============================================================================
def train_ra_exact(args, model, train_loader, val_loader, device):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    scaler = GradScaler('cuda')
    
    sample_ds = train_loader.dataset.datasets[0] if hasattr(train_loader.dataset, 'datasets') else train_loader.dataset
    phys = MultiParamPhysicsLoss({'Pr': sample_ds.params.get('Pr', 0.71), 'norm_params': sample_ds.norm_params},
                                 sample_ds.nano_props, dt=sample_ds.params.get('dt', 0.0001), 
                                 dx=1.0/(sample_ds.nx-1), dy=1.0/(sample_ds.ny-1)).to(device)

    # Use original 4-param normalization logic to keep GPU busy
    print("  [Setup] Calculating normalization weights...")
    norm_weights = calculate_physics_normalization(model, train_loader, phys, device)

    best_val = float('inf')
    warmup = int(args.epochs * 0.15)

    for epoch in range(args.epochs):
        curr_lambda = 0.05 if epoch >= warmup else 0.0
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inp, tgt, pd in pbar:
            inp, tgt = inp.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            # We feed all 4, but conceptually focusing on Ra's variations
            r, h, q, d = pd['Ra'].to(device), pd['Ha'].to(device), pd['Q'].to(device), pd['Da'].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                pred = model(inp, r, h, q, d)
                loss_mse = F.mse_loss(pred, tgt)
                if curr_lambda > 0:
                    p_l = phys.physics_residual_loss(inp[:, -1], pred, r, h, q, d)
                    loss_phys = sum(p_l[k].mean() * norm_weights[k] for k in p_l)
                else: loss_phys = torch.tensor(0.0, device=device)
                loss_total = loss_mse + curr_lambda * loss_phys

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix({'mse': f"{loss_mse.item():.6f}"})

        model.eval()
        v_mse = 0
        with torch.no_grad(), autocast('cuda'):
            for inp, tgt, pd in val_loader:
                pred = model(inp.to(device), pd['Ra'].to(device), pd['Ha'].to(device), pd['Q'].to(device), pd['Da'].to(device))
                v_mse += F.mse_loss(pred, tgt.to(device)).item()
        
        avg_v = v_mse / len(val_loader)
        print(f"  Val MSE: {avg_v:.6f}")
        scheduler.step(avg_v)
        if avg_v < best_val:
            best_val = avg_v
            torch.save(model.state_dict(), 'checkpoint_ra_v4_exact.pth')

# =============================================================================
# 2. Ra-Only Ultra Inference (Adam + L-BFGS)
# =============================================================================
def predict_ra_ultra_exact(model, physics_loss_fn, dataset, config, device, gt_p):
    model.eval()
    num_samples = min(config.get('num_inference_samples', 20), len(dataset))
    num_restarts = config.get('num_restarts', 4)
    indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)
    batch_input = torch.stack([dataset[i][0] for i in indices]).to(device)
    batch_target = torch.stack([dataset[i][1] for i in indices]).to(device)

    # FIX Ha, Q, Da to Ground Truth
    h_fixed = torch.full((num_restarts * num_samples,), gt_p['Ha'], device=device)
    q_fixed = torch.full((num_restarts * num_samples,), gt_p['Q'], device=device)
    d_fixed = torch.full((num_restarts * num_samples,), gt_p['Da'], device=device)

    # Only optimize Ra
    p_raw_ra = torch.randn((num_restarts, 1), device=device, requires_grad=True)
    optimizer_adam = optim.Adam([p_raw_ra], lr=config['inference_lr'])
    
    # Phase 1: Adam
    for step in range(config['inference_steps']):
        optimizer_adam.zero_grad()
        ra = 10**(np.log10(config['ra_min']) + (np.log10(config['ra_max']) - np.log10(config['ra_min'])) * torch.sigmoid(p_raw_ra))
        ra_e = ra.view(-1, 1).expand(-1, num_samples).reshape(-1)
        
        with autocast('cuda'):
            pred = model(batch_input.repeat(num_restarts, 1, 1, 1, 1), ra_e, h_fixed, q_fixed, d_fixed)
            l_data = (pred - batch_target.repeat(num_restarts, 1, 1, 1)).pow(2).view(num_restarts, num_samples, -1).mean(dim=(1, 2))
            
            # Physics loss can be unstable during inference, temporarily disable or set to 0
            # un, vn, tn = torch.chunk(batch_input.repeat(num_restarts, 1, 1, 1, 1)[:, -1], 4, 1)[:3]
            # unx, vnx, tnx, pnx = torch.chunk(pred, 4, 1)
            # l_ra_cons, _ = physics_loss_fn.ra_consistency_loss(un, vn, pnx, unx, vnx, tnx, ra_e, d_fixed, h_fixed)
            # l_ra_cons_r = l_ra_cons.view(num_restarts, num_samples).mean(dim=1)

            loss_total = (50.0 * l_data).mean() # Focus only on data loss for stability
        
        if torch.isnan(loss_total):
            print(f"  [Warning] NaN detected at step {step}. Breaking Adam phase.")
            break

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_([p_raw_ra], 0.1) # Even tighter clipping
        optimizer_adam.step()

        if step % 50 == 0: # Print less frequently
            with torch.no_grad():
                best_idx = torch.argmin(l_data)
                curr_ra = ra[best_idx].item()
                print(f"  [Adam Step {step:4d}] Loss: {loss_total.item():.6f} | Pred Ra: {curr_ra:.2e} | GT Ra: {gt_p['Ra']:.2e}")

    # Phase 2: L-BFGS Refinement
    with torch.no_grad():
        bi = torch.argmin(l_data).item()
        if torch.isnan(l_data[bi]):
             bi = 0 
        p_best_ra = p_raw_ra[bi:bi+1].detach().clone().requires_grad_(True)
    
    # L-BFGS can also be a source of NaN if LR is high
    opt_lbfgs = optim.LBFGS([p_best_ra], lr=0.01, max_iter=100, line_search_fn='strong_wolfe')
    lbfgs_step = [0]
    def closure():
        opt_lbfgs.zero_grad()
        ra_val = 10**(np.log10(config['ra_min']) + (np.log10(config['ra_max']) - np.log10(config['ra_min'])) * torch.sigmoid(p_best_ra))
        with autocast('cuda'):
            h_gt = torch.full((num_samples,), gt_p['Ha'], device=device)
            q_gt = torch.full((num_samples,), gt_p['Q'], device=device)
            d_gt = torch.full((num_samples,), gt_p['Da'], device=device)
            pred_l = model(batch_input, ra_val.expand(num_samples, 1).reshape(-1), h_gt, q_gt, d_gt)
            lt = (pred_l - batch_target).pow(2).mean() * 100.0
        
        if torch.isnan(lt):
            return torch.tensor(1e10, device=device, requires_grad=True)
            
        lt.backward()
        lbfgs_step[0] += 1
        if lbfgs_step[0] % 10 == 0:
            print(f"  [L-BFGS Step {lbfgs_step[0]:3d}] Loss: {lt.item():.6f} | Pred Ra: {ra_val.item():.2e} | GT Ra: {gt_p['Ra']:.2e}")
        return lt
    
    try:
        opt_lbfgs.step(closure)
    except Exception as e:
        print(f"  [L-BFGS Error] {e}")

    final_ra = 10**(np.log10(config['ra_min']) + (np.log10(config['ra_max']) - np.log10(config['ra_min'])) * torch.sigmoid(p_best_ra))
    return final_ra.item()

# ... (rest of the file)
    # Test on test set
    inf_config = {'inference_steps': 1500, 'inference_lr': 0.0005, 'ra_min': 100, 'ra_max': 1e8, 'num_restarts': 4}
    
    final_ra = 10**(np.log10(config['ra_min']) + (np.log10(config['ra_max']) - np.log10(config['ra_min'])) * torch.sigmoid(p_best_ra))
    return final_ra.item()

# =============================================================================
# 3. Main
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--base_fluid', default='EG')
    parser.add_argument('--inference_only', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Ra-Only EXACT V4.0 Initialized. Device: {device}")

    base_path = os.path.join('data', args.base_fluid)
    all_files = glob.glob(os.path.join(base_path, "**", "*.mat"), recursive=True)
    all_files = [f for f in sorted(all_files) if 'phi' not in f.lower()]
    random.seed(42); random.shuffle(all_files)

    train_files = all_files[:int(len(all_files)*0.8)]
    val_files = all_files[int(len(all_files)*0.8):int(len(all_files)*0.9)]
    test_files = all_files[int(len(all_files)*0.9):]
    
    cache_dir = f"cache_{args.base_fluid}"
    def get_loader(files):
        caches = [CachedSequenceDataset(preprocess_to_hdf5(f, cache_dir)) for f in files]
        return DataLoader(ConcatDataset(caches), batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    model = MultiParamSurrogateModel(hidden=256).to(device)
    if not args.inference_only:
        train_loader = get_loader(train_files); val_loader = get_loader(val_files)
        train_ra_exact(args, model, train_loader, val_loader, device)
    else:
        model.load_state_dict(torch.load('checkpoint_ra_v4_exact.pth', weights_only=True))
        print("Loaded trained model for inference.")

    # Test on test set
    inf_config = {'inference_steps': 1500, 'inference_lr': 0.005, 'ra_min': 100, 'ra_max': 1e8, 'num_restarts': 4}
    for f in test_files[:5]:
        ds_m = MatDataset(f, device=device)
        ds_s = CachedSequenceDataset(preprocess_to_hdf5(f, cache_dir), device=device)
        phys = MultiParamPhysicsLoss(ds_m.params, ds_m.nanofluid_props, dt=ds_m.params['dt'], 
                                    dx=1.0/(ds_m.nx-1), dy=1.0/(ds_m.ny-1)).to(device)
        res = predict_ra_ultra_exact(model, phys, ds_s, inf_config, device, ds_m.params)
        print(f"File: {os.path.basename(f)} | GT Ra: {ds_m.params['Ra']:.2e} | Pred Ra: {res:.2e} | Err: {abs(res-ds_m.params['Ra'])/ds_m.params['Ra']*100:.2f}%")

if __name__ == '__main__':
    main()
