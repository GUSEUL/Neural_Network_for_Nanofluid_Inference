import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import argparse
import random
import glob
import time
from tqdm import tqdm

from data import MatDataset
from models import MultiParamSurrogateModel
from train_and_infer_v4 import (
    MultiParamPhysicsLoss,
    CachedSequenceDataset,
    preprocess_to_hdf5,
    predict_multi_params_ultra
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoint_best_EG.pth', help='Model checkpoint path')
    parser.add_argument('--base_fluid', default='EG')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--steps', type=int, default=1500, help='Adam optimization steps')
    parser.add_argument('--lbfgs_steps', type=int, default=100, help='L-BFGS max iterations')
    parser.add_argument('--restarts', type=int, default=4, help='Number of restarts')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of files')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference Mode Initialized. Device: {device}")

    # 1. File Selection
    base_path = os.path.join(args.data_root, args.base_fluid)
    all_files = glob.glob(os.path.join(base_path, "**", "*.mat"), recursive=True)
    all_files = [f for f in sorted(all_files) if 'phi' not in f.lower()]

    random.seed(42)
    random.shuffle(all_files)

    n = len(all_files)
    test_files = all_files[int(n*0.9):]

    if args.limit:
        test_files = test_files[:args.limit]

    if not test_files:
        print("No test files found.")
        return

    print(f"Total Test Files: {len(test_files)}")

    # 2. Load Model
    model = MultiParamSurrogateModel(hidden=256).to(device)
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return
    model.eval()

    # 3. Config
    inf_config = {
        'inference_steps': args.steps,
        'inference_lr': 0.005,
        'lbfgs_steps': args.lbfgs_steps,
        'ra_min': 100, 'ra_max': 1e8, 'ha_min': 0, 'ha_max': 100,
        'q_min': -10, 'q_max': 10, 'da_min': 0.001, 'da_max': 0.15,
        'num_inference_samples': 20
    }
    norm_w = {'continuity': 1.0, 'momentum_x': 1.0, 'momentum_y': 3.0, 'energy': 3.0}
    cache_dir = f"cache_{args.base_fluid}"

    # Results stats
    all_errors = {'Ra': [], 'Ha': [], 'Q': [], 'Da': []}
    start_time_all = time.time()

    # 4. Processing
    for idx, target_f in enumerate(test_files):
        print(f"\n" + "="*80)
        print(f"[{idx+1}/{len(test_files)}] Processing: {os.path.basename(target_f)}")
        print("="*80)

        ds_mat = MatDataset(target_f, device=device)
        print(f"GROUND TRUTH: Ra={ds_mat.params['Ra']:.2e}, Ha={ds_mat.params['Ha']:.2f}, Q={ds_mat.params['Q']:.2f}, Da={ds_mat.params['Da']:.4f}")

        cache_path = preprocess_to_hdf5(target_f, cache_dir)
        ds_seq = CachedSequenceDataset(cache_path, device=device)

        phys = MultiParamPhysicsLoss(ds_mat.params, ds_mat.nanofluid_props,
                                    dt=ds_mat.params['dt'], dx=1.0/(ds_mat.nx-1), dy=1.0/(ds_mat.ny-1)).to(device)

        # Starting inference
        print(f"Starting inference (Adam: {args.steps}, L-BFGS: {args.lbfgs_steps})...")
        log_name = os.path.basename(target_f).split('.')[0]
        pred_p = predict_multi_params_ultra(model, phys, ds_seq, inf_config, device,
                                           norm_weights=norm_w, num_restarts=args.restarts,
                                           log_prefix=log_name)

        # Results Summary
        print(f"\nResults for {os.path.basename(target_f)}:")
        for p in ['Ra', 'Ha', 'Q', 'Da']:
            true_v = ds_mat.params.get(p)
            pred_v = pred_p[p]
            err = abs(true_v - pred_v) / (abs(true_v) + 1e-8) * 100
            all_errors[p].append(err)
            print(f"  {p:2s}: True={true_v:10.4f}, Pred={pred_v:10.4f}, Err={err:6.2f}%")

    end_time_all = time.time()
    print(f"\n" + "#"*80)
    print(f"FINAL SUMMARY ({len(test_files)} files, Total time: {end_time_all - start_time_all:.1f}s)")
    print("#"*80)
    for p in ['Ra', 'Ha', 'Q', 'Da']:
        if all_errors[p]:
            avg_err = np.mean(all_errors[p])
            max_err = np.max(all_errors[p])
            print(f"  {p:2s} | Average Error: {avg_err:6.2f}% | Max Error: {max_err:6.2f}%")
    print("#"*80)

if __name__ == '__main__':
    main()
