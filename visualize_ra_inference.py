import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
import numpy as np
import os
import glob
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

from data import MatDataset
from models import MultiParamSurrogateModel
from train_and_infer_v4 import (
    preprocess_to_hdf5, CachedSequenceDataset
)

# =============================================================================
# 1. Ra Inference (Data Loss Only) & Continuous Rollout
# =============================================================================
def predict_ra_and_rollout(model, dataset, config, device, gt_p, rollout_len=100):
    model.eval()
    num_inference_samples = config.get('num_inference_samples', 20)
    num_restarts = config.get('num_restarts', 4)
    
    # 1. Optimization phase (Data Loss Only for stability)
    indices = np.linspace(0, len(dataset)-1, num_inference_samples, dtype=int)
    batch_input = torch.stack([dataset[i][0] for i in indices]).to(device)
    batch_target = torch.stack([dataset[i][1] for i in indices]).to(device)

    h_fixed = torch.full((num_restarts * num_inference_samples,), gt_p['Ha'], device=device)
    q_fixed = torch.full((num_restarts * num_inference_samples,), gt_p['Q'], device=device)
    d_fixed = torch.full((num_restarts * num_inference_samples,), gt_p['Da'], device=device)

    p_raw_ra = torch.randn((num_restarts, 1), device=device, requires_grad=True)
    optimizer_adam = optim.Adam([p_raw_ra], lr=config['inference_lr'])
    
    ra_history = []
    print("  [Step 1] Optimizing Ra (Data Loss Matching)...")
    for step in range(config['inference_steps']):
        optimizer_adam.zero_grad()
        ra = 10**(np.log10(config['ra_min']) + (np.log10(config['ra_max']) - np.log10(config['ra_min'])) * torch.sigmoid(p_raw_ra))
        ra_e = ra.view(-1, 1).expand(-1, num_inference_samples).reshape(-1)
        
        with autocast('cuda'):
            pred = model(batch_input.repeat(num_restarts, 1, 1, 1, 1), ra_e, h_fixed, q_fixed, d_fixed)
            l_data = (pred - batch_target.repeat(num_restarts, 1, 1, 1)).pow(2).view(num_restarts, num_inference_samples, -1).mean(dim=(1, 2))
            loss_total = (50.0 * l_data).mean()
        
        if torch.isnan(loss_total): break
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_([p_raw_ra], 0.1)
        optimizer_adam.step()

        best_idx = torch.argmin(l_data).item()
        curr_ra_val = ra[best_idx].item()
        ra_history.append(curr_ra_val)
        if step % 50 == 0:
            print(f"    [Step {step:4d}] Loss: {loss_total.item():.6f} | Pred Ra: {curr_ra_val:.2e}")

    # Final Ra
    bi = torch.argmin(l_data).item()
    final_ra_val = ra[bi].detach()
    final_ra_item = final_ra_val.item()

    # 2. Continuous Rollout & High-Res Interpolation
    print(f"  [Step 2] Generating High-Res Rollout ({rollout_len} frames)...")
    target_res = 256 # Higher resolution for visualization
    
    with torch.no_grad(), autocast('cuda'):
        current_seq = dataset[0][0].unsqueeze(0).to(device)
        preds_hi = []
        gts_hi = []
        
        ra_param = final_ra_val.view(1)
        h_param = torch.tensor([gt_p['Ha']], device=device)
        q_param = torch.tensor([gt_p['Q']], device=device)
        d_param = torch.tensor([gt_p['Da']], device=device)

        for i in range(rollout_len):
            # Predict
            out = model(current_seq, ra_param, h_param, q_param, d_param) # [1, 4, 42, 42]
            
            # Upsample Prediction Only (Bicubic)
            out_hi = F.interpolate(out, size=(target_res, target_res), mode='bicubic', align_corners=True)
            preds_hi.append(out_hi.squeeze(0).cpu().numpy())
            
            # Keep GT at Original Resolution (No interpolation)
            if i < len(dataset):
                gts_hi.append(dataset[i][1].cpu().numpy()) # Store original 42x42
            
            # Autoregressive update
            current_seq = torch.cat([current_seq[:, 1:], out.unsqueeze(1)], dim=1)

    return final_ra_item, ra_history, np.array(gts_hi), np.array(preds_hi)

# =============================================================================
# 2. Enhanced Visualization (GT Orig vs Pred High-Res)
# =============================================================================
def save_enhanced_visualizations(filename, gt_ra, pred_ra, ra_history, gt_seq, pred_seq, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(filename))[0]

    # 1. Convergence Plot
    plt.figure(figsize=(10, 5))
    plt.plot(ra_history, label='Predicted Ra', color='blue', lw=2)
    plt.axhline(y=gt_ra, color='red', linestyle='--', label='Ground Truth Ra', alpha=0.7)
    plt.yscale('log')
    plt.title(f"Ra Convergence Path (Data Loss Only)\nGT={gt_ra:.2e} | Pred={pred_ra:.2e}")
    plt.xlabel("Optimization Steps")
    plt.ylabel("Rayleigh Number (Ra)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_convergence.png"), dpi=150)
    plt.close()

    # 2. Comparison Animation (GT Original vs Pred High-Res)
    num_frames = len(pred_seq)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    titles = ['u-velocity', 'v-velocity', 'Temperature', 'Pressure']
    plt.subplots_adjust(top=0.88, bottom=0.05, left=0.03, right=0.97, hspace=0.15, wspace=0.1)

    im_objs = []
    for row in range(2):
        row_objs = []
        for col in range(4):
            # Row 0: GT (42x42), Row 1: Pred (256x256)
            res = 42 if row == 0 else 256
            im = axes[row, col].imshow(np.zeros((res, res)), cmap='jet', origin='lower', interpolation='none')
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0: axes[row, col].set_title(f"GT {titles[col]} (Original 42x42)", fontsize=11)
            else: axes[row, col].set_title(f"Continuous Pred {titles[col]} (High-Res 256x256)", fontsize=11, color='blue', fontweight='bold')
            row_objs.append(im)
        im_objs.append(row_objs)

    def update(frame):
        for i in range(4):
            if frame < len(gt_seq):
                im_objs[0][i].set_data(gt_seq[frame, i])
                im_objs[0][i].set_clim(gt_seq[frame, i].min(), gt_seq[frame, i].max())
            im_objs[1][i].set_data(pred_seq[frame, i])
            im_objs[1][i].set_clim(pred_seq[frame, i].min(), pred_seq[frame, i].max())
            
        fig.suptitle(f"Continuous Flow Synthesis (256x256 Bicubic) | Frame {frame:03d}/{num_frames-1}\nRa GT: {gt_ra:.2e} | Predicted: {pred_ra:.2e}", fontsize=18)
        return [item for sublist in im_objs for item in sublist]

    anim = FuncAnimation(fig, update, frames=num_frames, interval=80, blit=True)
    anim.save(os.path.join(output_dir, f"{base_name}_high_res_motion.gif"), writer=PillowWriter(fps=12))
    plt.close()
    print(f"  [Saved] {base_name}_convergence.png and {base_name}_high_res_motion.gif")

# =============================================================================
# 3. Main
# =============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'checkpoint_ra_v4_exact.pth'
    output_dir = 'inference_results_high_res'
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return

    model = MultiParamSurrogateModel(hidden=256).to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    print("Model loaded.")

    base_path = os.path.join('data', 'EG')
    all_files = glob.glob(os.path.join(base_path, "**", "*.mat"), recursive=True)
    all_files = [f for f in sorted(all_files) if 'phi' not in f.lower()]
    random.seed(42); random.shuffle(all_files)
    test_files = all_files[int(len(all_files)*0.9):int(len(all_files)*0.9) + 10]

    inf_config = {
        'inference_steps': 1000, 
        'inference_lr': 0.0005, 
        'ra_min': 100, 
        'ra_max': 1e8, 
        'num_restarts': 4,
        'num_inference_samples': 20
    }

    cache_dir = "cache_EG"
    print(f"\n>>> Starting High-Res Visualization (Bicubic Upsampling) for {len(test_files)} samples...")
    
    pbar_files = tqdm(test_files, desc="Total Progress")
    for f in pbar_files:
        pbar_files.set_postfix({"file": os.path.basename(f)})
        ds_m = MatDataset(f, device=device)
        ds_s = CachedSequenceDataset(preprocess_to_hdf5(f, cache_dir), device=device)
        
        res_ra, history, gt_seq, pred_seq = predict_ra_and_rollout(
            model, ds_s, inf_config, device, ds_m.params, rollout_len=100
        )
        
        save_enhanced_visualizations(f, ds_m.params['Ra'], res_ra, history, gt_seq, pred_seq, output_dir)
        err = abs(res_ra-ds_m.params['Ra'])/ds_m.params['Ra']*100
        print(f"  [Result] {os.path.basename(f)} | Err: {err:.2f}%")

if __name__ == '__main__':
    main()
