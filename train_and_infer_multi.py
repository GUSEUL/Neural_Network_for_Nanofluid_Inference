"""
Multi-Parameter Inverse Problem Solver
=======================================

Simultaneously infers Ra, Ha, Q, and Da from flow field data using a
physics-informed neural network approach.

Governing Equations (Nanofluid in Porous Media with MHD):
---------------------------------------------------------
  Continuity:
    dU/dx + dV/dy = 0

  X-Momentum:
    dU/dt + U·dU/dx + V·dU/dy = -dP/dx + ν·Pr·∇²U - (ν·Pr/Da)·U

  Y-Momentum:
    dV/dt + U·dV/dx + V·dV/dy = -dP/dy + ν·Pr·∇²V
                                 + β·Ra·Pr·θ - (ν·Pr/Da)·V - σ·ρ·Ha²·Pr·V

  Energy:
    dθ/dt + U·dθ/dx + V·dθ/dy = α·∇²θ + (ρCp_f/ρCp_thnf)·Q·θ

Data Strategy:
--------------
  - All subfolders (EG_Da, EG_Ha, EG_Q, EG_Ra) under a base fluid directory
    are pooled together.
  - 80% train / 10% validation / 10% test split.
  - Each .mat file contains the ground truth values for ALL parameters
    (Ra, Ha, Q, Da), even though only one parameter varies per subfolder.

Inference Strategy:
-------------------
  For each test file, all 4 parameters are optimized simultaneously by
  minimizing a combined loss:
    L = L_data + λ_phys * L_physics + Σ λ_consistency_i * L_consistency_i

How to Run:
-----------
# Basic training + inference
python train_and_infer_multi.py --base_fluid EG

# Skip training (use pre-trained model)
python train_and_infer_multi.py --base_fluid EG --skip_train

# Multi-timestep inference
python train_and_infer_multi.py --base_fluid EG --skip_train --multi_timestep

# Use stable frame for inference
python train_and_infer_multi.py --base_fluid EG --skip_train --stable_frame_idx 100

# Adjust training epochs
python train_and_infer_multi.py --base_fluid EG --epochs 200

# Enable FiLM conditioning
python train_and_infer_multi.py --base_fluid EG --use_film

Arguments:
----------
--base_fluid       : Base fluid name (e.g., EG, Water, Kerosene)
--data_root        : Root directory for data files (default: data)
--epochs           : Number of training epochs (default: 100)
--results_dir      : Output directory for results (default: multi_inference_results)
--skip_train       : Skip training phase and load existing model
--inference_lr     : Learning rate for inference optimization (default: 0.005)
--inference_steps  : Number of optimization steps for inference (default: 1500)
--multi_timestep   : Use multiple timesteps for more stable inference
--use_film         : Enable FiLM conditioning for stronger parameter dependency
--test_epochs      : Override inference steps for test
--stable_frame_idx : Use specific stable frame index for inference
--batch_size       : Training batch size (default: 16)
--checkpoint_interval : Save checkpoint every N epochs (default: 20)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import os
import json
import glob
import argparse
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
import traceback

from data import MatDataset
from models import STNNN


# =============================================================================
# Discord Notification
# =============================================================================
DISCORD_WEBHOOK_URL = "https://discordapp.com/api/webhooks/1473922778678693970/5demVGFqj3qSrznYADoI45BNICjBHCahTZlc9k44ZvTcPuhyA8C7MmHu8sCiekOAFjUc"


def send_discord_error(error_msg, title="Error Traceback"):
    """Send error message to Discord via webhook."""
    if len(error_msg) > 3900:
        error_msg = error_msg[:3900] + "\n... (truncated)"
    data = {
        "content": "**[train_and_infer_multi.py] Error occurred during execution!**",
        "embeds": [{
            "title": title,
            "description": f"```python\n{error_msg}\n```",
            "color": 16711680
        }]
    }
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=data, timeout=10)
        if response.status_code == 204:
            print("[Discord] Error notification sent successfully")
        else:
            print(f"[Discord] Failed to send notification: {response.status_code}")
    except Exception as e:
        print(f"[Discord] Failed to send notification: {e}")


def send_discord_success(message, results=None):
    """Send success message to Discord via webhook."""
    results_text = ""
    if results:
        for r in results[:10]:  # Limit to first 10 results
            if r.get('true_values') is not None:
                results_text += f"\n**{r['file']}**:"
                for param, true_v in r['true_values'].items():
                    pred_v = r['predicted_values'].get(param, 'N/A')
                    err = r['percent_errors'].get(param, 'N/A')
                    if isinstance(pred_v, (int, float)) and isinstance(err, (int, float)):
                        results_text += f"\n  {param}: True={true_v:.4f}, Pred={pred_v:.4f}, Error={err:.2f}%"
    
    # Truncate if too long
    if len(results_text) > 3500:
        results_text = results_text[:3500] + "\n... (truncated)"
    
    data = {
        "content": "**[train_and_infer_multi.py] Execution completed successfully!**",
        "embeds": [{
            "title": "Multi-Parameter Inference Summary",
            "description": f"{message}{results_text}",
            "color": 65280
        }]
    }
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=data, timeout=10)
        if response.status_code == 204:
            print("[Discord] Success notification sent")
    except Exception as e:
        print(f"[Discord] Failed to send notification: {e}")


# =============================================================================
# Component 1: Multi-Parameter Physics Loss
# =============================================================================
class MultiParamPhysicsLoss(nn.Module):
    """
    Unified physics loss for simultaneous inference of Ra, Ha, Q, Da.
    
    Each parameter appears in specific equations:
      - Da: x-momentum (porous drag), y-momentum (porous drag)
      - Ha: y-momentum (magnetic Lorentz force)
      - Ra: y-momentum (buoyancy)
      - Q:  energy equation (heat source)
    
    During training, all parameters are known (from the .mat file).
    During inference, all four are optimized simultaneously.
    """
    
    def __init__(self, params, nanofluid_props=None, dt=0.0001, dx=1.0, dy=1.0):
        super().__init__()
        self.Pr = params['Pr']
        self.Ra = params.get('Ra', 1e4)
        self.Ha = params.get('Ha', 0.0)
        self.Da = params.get('Da', 1e-3)
        self.Q = params.get('Q', 0.0)
        
        # Nanofluid property ratios
        if nanofluid_props is not None:
            self.nu_thnf_ratio = nanofluid_props['nu_thnf_ratio']
            self.sigma_thnf_ratio = nanofluid_props['sigma_thnf_ratio']
            self.rho_f_thnf_ratio = nanofluid_props['rho_f_thnf_ratio']
            self.beta_thnf_ratio = nanofluid_props['beta_thnf_ratio']
            self.alpha_thnf_ratio = nanofluid_props['alpha_thnf_ratio']
            self.rhocp_f_thnf_ratio = nanofluid_props['rhocp_f_thnf_ratio']
        else:
            self.nu_thnf_ratio = 1.0
            self.sigma_thnf_ratio = 1.0
            self.rho_f_thnf_ratio = 1.0
            self.beta_thnf_ratio = 1.0
            self.alpha_thnf_ratio = 1.0
            self.rhocp_f_thnf_ratio = 1.0

        self.dt, self.dx, self.dy = dt, dx, dy
        
        # Loss weights
        self.w_continuity = 1.0
        self.w_momentum_x = 1.0
        self.w_momentum_y = 1.0
        self.w_energy = 1.0

    # ----- Derivative Utilities -----
    def compute_derivatives(self, field):
        """Compute spatial derivatives using central difference scheme."""
        dfdx = torch.gradient(field, dim=-1)[0] / self.dx
        dfdy = torch.gradient(field, dim=-2)[0] / self.dy
        dfdx = torch.clamp(dfdx, -100, 100)
        dfdy = torch.clamp(dfdy, -100, 100)
        d2fdx2 = torch.gradient(dfdx, dim=-1)[0] / self.dx
        d2fdy2 = torch.gradient(dfdy, dim=-2)[0] / self.dy
        d2fdx2 = torch.clamp(d2fdx2, -1000, 1000)
        d2fdy2 = torch.clamp(d2fdy2, -1000, 1000)
        return {'dx': dfdx, 'dy': dfdy, 'dxx': d2fdx2, 'dyy': d2fdy2}

    def compute_time_derivative(self, f_now, f_next):
        return (f_next - f_now) / self.dt

    # ----- Residual Equations -----
    def continuity_residual(self, U, V):
        """dU/dx + dV/dy = 0"""
        return self.compute_derivatives(U)['dx'] + self.compute_derivatives(V)['dy']

    def momentum_x_residual(self, U_now, V_now, P_next, U_next, V_next,
                            da_value, steady_state=False):
        """
        X-Momentum: dU/dt + U·dU/dx + V·dU/dy = -dP/dx + ν·Pr·∇²U - (ν·Pr/Da)·U
        
        Da appears in the porous drag term.
        """
        if steady_state:
            dUdt = torch.zeros_like(U_next)
        else:
            dUdt = self.compute_time_derivative(U_now, U_next)
        
        U_derivs = self.compute_derivatives(U_next)
        P_derivs = self.compute_derivatives(P_next)
        
        convection = U_next * U_derivs['dx'] + V_now * U_derivs['dy']
        pressure_grad = -P_derivs['dx']
        viscous = self.nu_thnf_ratio * self.Pr * (U_derivs['dxx'] + U_derivs['dyy'])
        porous_drag = -(self.nu_thnf_ratio * self.Pr / da_value) * U_next
        
        return (dUdt + convection) - (pressure_grad + viscous + porous_drag)

    def momentum_y_residual(self, U_now, V_now, P_next, U_next, V_next, theta_next,
                            ra_value, ha_value, da_value, steady_state=False):
        """
        Y-Momentum: dV/dt + U·dV/dx + V·dV/dy = -dP/dy + ν·Pr·∇²V
                     + β·Ra·Pr·θ - (ν·Pr/Da)·V - σ·ρ·Ha²·Pr·V
        
        Ra, Ha, Da all appear here.
        """
        if steady_state:
            dVdt = torch.zeros_like(V_next)
        else:
            dVdt = self.compute_time_derivative(V_now, V_next)
        
        V_derivs = self.compute_derivatives(V_next)
        P_derivs = self.compute_derivatives(P_next)
        
        convection = U_now * V_derivs['dx'] + V_next * V_derivs['dy']
        pressure_grad = -P_derivs['dy']
        viscous = self.nu_thnf_ratio * self.Pr * (V_derivs['dxx'] + V_derivs['dyy'])
        buoyancy = self.beta_thnf_ratio * ra_value * self.Pr * theta_next
        porous_drag = -(self.nu_thnf_ratio * self.Pr / da_value) * V_next
        magnetic = -(self.sigma_thnf_ratio * self.rho_f_thnf_ratio * (ha_value ** 2) * self.Pr) * V_next
        
        return (dVdt + convection) - (pressure_grad + viscous + buoyancy + porous_drag + magnetic)

    def energy_residual(self, U_now, V_now, theta_now, theta_next,
                        q_value, steady_state=False):
        """
        Energy: dθ/dt + U·dθ/dx + V·dθ/dy = α·∇²θ + (ρCp_f/ρCp_thnf)·Q·θ
        
        Q appears in the heat source term.
        """
        if steady_state:
            dthetadt = torch.zeros_like(theta_next)
        else:
            dthetadt = self.compute_time_derivative(theta_now, theta_next)
        
        theta_derivs = self.compute_derivatives(theta_next)
        convection = U_now * theta_derivs['dx'] + V_now * theta_derivs['dy']
        diffusion = self.alpha_thnf_ratio * (theta_derivs['dxx'] + theta_derivs['dyy'])
        heat_source = self.rhocp_f_thnf_ratio * q_value * theta_next
        
        return (dthetadt + convection) - (diffusion + heat_source)

    # ----- Parameter Inference from Equations -----
    def infer_da_from_momentum_x(self, U_now, V_now, P_next, U_next, V_next, steady_state=False):
        """
        Infer Da from x-momentum:
            (ν·Pr/Da)·U = -(dU/dt + convection - pressure_grad - viscous)
            Da = (ν·Pr·U) / (pressure_grad + viscous - dU/dt - convection)
        """
        if steady_state:
            dUdt = torch.zeros_like(U_next)
        else:
            dUdt = self.compute_time_derivative(U_now, U_next)
        
        U_derivs = self.compute_derivatives(U_next)
        P_derivs = self.compute_derivatives(P_next)
        
        convection = U_next * U_derivs['dx'] + V_now * U_derivs['dy']
        pressure_grad = -P_derivs['dx']
        viscous = self.nu_thnf_ratio * self.Pr * (U_derivs['dxx'] + U_derivs['dyy'])
        
        # residual_without_da = (dUdt + convection) - (pressure_grad + viscous)
        # Should equal porous_drag = -(nu*Pr/Da)*U
        # So Da = -(nu*Pr*U) / residual_without_da
        residual_without_da = (dUdt + convection) - (pressure_grad + viscous)
        
        numerator = -(self.nu_thnf_ratio * self.Pr * U_next)
        
        if steady_state:
            denom_mag = torch.abs(residual_without_da)
            max_denom = torch.max(denom_mag)
            threshold = max(1e-10, max_denom.item() * 1e-6) if max_denom.item() > 0 else 1e-10
            safe_denom = torch.where(denom_mag < threshold,
                                     torch.sign(residual_without_da) * threshold,
                                     residual_without_da)
        else:
            safe_denom = torch.where(torch.abs(residual_without_da) < 1e-8,
                                     torch.sign(residual_without_da) * 1e-8 + 1e-10,
                                     residual_without_da)
        
        da_inferred = numerator / safe_denom
        da_inferred = torch.clamp(da_inferred, min=0.001, max=1.0)
        return da_inferred

    def infer_ra_from_momentum_y(self, U_now, V_now, P_next, U_next, V_next, theta_next,
                                  da_value=None, ha_value=None, steady_state=False):
        """
        Infer Ra from y-momentum (with known/guessed Da, Ha):
            buoyancy = residual_without_Ra
            Ra = residual_without_Ra / (β · Pr · θ)
        """
        if da_value is None:
            da_value = self.Da
        if ha_value is None:
            ha_value = self.Ha
        
        if steady_state:
            dVdt = torch.zeros_like(V_next)
        else:
            dVdt = self.compute_time_derivative(V_now, V_next)
        
        V_derivs = self.compute_derivatives(V_next)
        P_derivs = self.compute_derivatives(P_next)
        
        convection = U_now * V_derivs['dx'] + V_next * V_derivs['dy']
        pressure_grad = -P_derivs['dy']
        viscous = self.nu_thnf_ratio * self.Pr * (V_derivs['dxx'] + V_derivs['dyy'])
        porous_drag = -(self.nu_thnf_ratio * self.Pr / da_value) * V_next
        magnetic = -(self.sigma_thnf_ratio * self.rho_f_thnf_ratio * (ha_value ** 2) * self.Pr) * V_next
        
        residual_without_ra = (dVdt + convection) - (pressure_grad + viscous + porous_drag + magnetic)
        
        denominator = self.beta_thnf_ratio * self.Pr * theta_next
        
        if steady_state:
            denom_mag = torch.abs(denominator)
            max_denom = torch.max(denom_mag)
            threshold = max(1e-10, max_denom.item() * 1e-6) if max_denom.item() > 0 else 1e-10
            safe_denom = torch.where(denom_mag < threshold,
                                     torch.sign(denominator) * threshold,
                                     denominator)
        else:
            safe_denom = torch.where(torch.abs(denominator) < 1e-8,
                                     torch.sign(denominator) * 1e-8 + 1e-10,
                                     denominator)
        
        ra_inferred = residual_without_ra / safe_denom
        ra_inferred = torch.clamp(ra_inferred, min=1.0, max=1e8)
        return ra_inferred

    def infer_ha_from_momentum_y(self, U_now, V_now, P_next, U_next, V_next, theta_next,
                                  ra_value=None, da_value=None, steady_state=False):
        """
        Infer Ha from y-momentum (with known/guessed Da, Ra):
            magnetic = residual_without_Ha
            σ·ρ·Ha²·Pr·V = -residual_without_Ha
            Ha² = residual_without_Ha / (σ·ρ·Pr·V)
        """
        if ra_value is None:
            ra_value = self.Ra
        if da_value is None:
            da_value = self.Da
        
        if steady_state:
            dVdt = torch.zeros_like(V_next)
        else:
            dVdt = self.compute_time_derivative(V_now, V_next)
        
        V_derivs = self.compute_derivatives(V_next)
        P_derivs = self.compute_derivatives(P_next)
        
        convection = U_now * V_derivs['dx'] + V_next * V_derivs['dy']
        pressure_grad = -P_derivs['dy']
        viscous = self.nu_thnf_ratio * self.Pr * (V_derivs['dxx'] + V_derivs['dyy'])
        buoyancy = self.beta_thnf_ratio * ra_value * self.Pr * theta_next
        porous_drag = -(self.nu_thnf_ratio * self.Pr / da_value) * V_next
        
        residual_without_ha = (dVdt + convection) - (pressure_grad + viscous + buoyancy + porous_drag)
        
        denominator = self.sigma_thnf_ratio * self.rho_f_thnf_ratio * self.Pr * V_next
        
        safe_denom = torch.where(torch.abs(denominator) < 1e-6,
                                 torch.full_like(denominator, 1e-6),
                                 denominator)
        
        ha_squared = residual_without_ha / safe_denom
        ha_squared_clamped = torch.clamp(ha_squared, min=0.0, max=1e4)
        ha_inferred = torch.sqrt(ha_squared_clamped)
        return ha_inferred

    def infer_q_from_energy(self, U_now, V_now, theta_now, theta_next, steady_state=False):
        """
        Infer Q from energy equation:
            heat_source = residual_without_Q
            (ρCp_f/ρCp_thnf)·Q·θ = residual_without_Q
            Q = residual_without_Q / ((ρCp_f/ρCp_thnf)·θ)
        """
        if steady_state:
            dthetadt = torch.zeros_like(theta_next)
        else:
            dthetadt = self.compute_time_derivative(theta_now, theta_next)
        
        theta_derivs = self.compute_derivatives(theta_next)
        convection = U_now * theta_derivs['dx'] + V_now * theta_derivs['dy']
        diffusion = self.alpha_thnf_ratio * (theta_derivs['dxx'] + theta_derivs['dyy'])
        
        residual_without_q = (dthetadt + convection) - diffusion
        
        denominator = self.rhocp_f_thnf_ratio * theta_next
        
        if steady_state:
            denom_mag = torch.abs(denominator)
            max_denom = torch.max(denom_mag)
            threshold = max(1e-10, max_denom.item() * 1e-6) if max_denom.item() > 0 else 1e-10
            safe_denom = torch.where(denom_mag < threshold,
                                     torch.sign(denominator) * threshold,
                                     denominator)
        else:
            safe_denom = torch.where(torch.abs(denominator) < 1e-8,
                                     torch.sign(denominator) * 1e-8 + 1e-10,
                                     denominator)
        
        q_inferred = residual_without_q / safe_denom
        q_inferred = torch.clamp(q_inferred, min=-20.0, max=20.0)
        return q_inferred

    # ----- Consistency Losses -----
    def da_consistency_loss(self, U_now, V_now, P_next, U_next, V_next, da_guess, steady_state=False):
        """MSE between inferred Da (from x-momentum) and guessed Da."""
        da_inferred = self.infer_da_from_momentum_x(
            U_now, V_now, P_next, U_next, V_next, steady_state=steady_state
        )
        da_inferred_mean = da_inferred.mean()
        loss = torch.mean((da_inferred - da_guess) ** 2)
        return loss, da_inferred_mean

    def ra_consistency_loss(self, U_now, V_now, P_next, U_next, V_next, theta_next, ra_guess,
                            da_value=None, ha_value=None, steady_state=False):
        """Relative-error consistency loss for Ra (large-scale values)."""
        ra_inferred = self.infer_ra_from_momentum_y(
            U_now, V_now, P_next, U_next, V_next, theta_next,
            da_value=da_value, ha_value=ha_value, steady_state=steady_state
        )
        ra_inferred_mean = ra_inferred.mean()
        loss = torch.mean(((ra_inferred - ra_guess) / (torch.abs(ra_guess) + 1e-6)) ** 2)
        return loss, ra_inferred_mean

    def ha_consistency_loss(self, U_now, V_now, P_next, U_next, V_next, theta_next, ha_guess,
                            ra_value=None, da_value=None, steady_state=False):
        """MSE between inferred Ha and guessed Ha."""
        ha_inferred = self.infer_ha_from_momentum_y(
            U_now, V_now, P_next, U_next, V_next, theta_next,
            ra_value=ra_value, da_value=da_value, steady_state=steady_state
        )
        ha_inferred_mean = ha_inferred.mean()
        loss = torch.mean((ha_inferred - ha_guess) ** 2)
        return loss, ha_inferred_mean

    def q_consistency_loss(self, U_now, V_now, theta_now, theta_next, q_guess, steady_state=False):
        """MSE between inferred Q and guessed Q."""
        q_inferred = self.infer_q_from_energy(
            U_now, V_now, theta_now, theta_next, steady_state=steady_state
        )
        q_inferred_mean = q_inferred.mean()
        loss = torch.mean((q_inferred - q_guess) ** 2)
        return loss, q_inferred_mean

    # ----- Physics Residual Loss (with all params as variables) -----
    def physics_residual_loss(self, input_state, prediction, ra_value, ha_value, q_value, da_value,
                              steady_state=False):
        """
        Compute all physics residuals with the given parameter values.
        
        Returns dict of individual losses and total.
        """
        U_now, V_now, T_now, P_now = torch.chunk(input_state, 4, 1)
        U_next, V_next, T_next, P_next = torch.chunk(prediction, 4, 1)
        
        continuity_res = self.continuity_residual(U_next, V_next)
        
        momentum_x_res = self.momentum_x_residual(
            U_now, V_now, P_next, U_next, V_next,
            da_value=da_value, steady_state=steady_state
        )
        
        momentum_y_res = self.momentum_y_residual(
            U_now, V_now, P_next, U_next, V_next, T_next,
            ra_value=ra_value, ha_value=ha_value, da_value=da_value,
            steady_state=steady_state
        )
        
        energy_res = self.energy_residual(
            U_now, V_now, T_now, T_next,
            q_value=q_value, steady_state=steady_state
        )
        
        loss_c = torch.mean(continuity_res ** 2)
        loss_mx = torch.mean(momentum_x_res ** 2)
        loss_my = torch.mean(momentum_y_res ** 2)
        loss_e = torch.mean(energy_res ** 2)
        
        return {
            'continuity': loss_c,
            'momentum_x': loss_mx,
            'momentum_y': loss_my,
            'energy': loss_e,
            'total': loss_c + loss_mx + loss_my + loss_e
        }

    def forward(self, input_state, prediction, ra_value=None, ha_value=None,
                q_value=None, da_value=None, validation_mode=False, steady_state=False):
        """
        Compute physics loss. Uses stored ground truth values if args are None.
        """
        if ra_value is None:
            ra_value = self.Ra
        if ha_value is None:
            ha_value = self.Ha
        if q_value is None:
            q_value = self.Q
        if da_value is None:
            da_value = self.Da
        
        losses = self.physics_residual_loss(
            input_state, prediction, ra_value, ha_value, q_value, da_value,
            steady_state=steady_state
        )
        
        if validation_mode:
            return losses
        return losses['total']


# =============================================================================
# Component 2: Surrogate Model with Multi-Parameter FiLM Conditioning
# =============================================================================
class SequenceSTNNN(STNNN):
    """STNNN that processes sequence inputs."""
    
    def forward(self, x_sequence):
        B, S, C, H, W = x_sequence.shape
        x_reshaped = x_sequence.view(B * S, C, H, W)
        z_reshaped = self.enc(x_reshaped)
        _, hidden_C, hidden_H, hidden_W = z_reshaped.shape
        z_sequence = z_reshaped.view(B, S, hidden_C, hidden_H, hidden_W)
        lstm_out_sequence, _ = self.conv_lstm(z_sequence)
        last_step_output = lstm_out_sequence[:, -1, :, :, :]
        z = self.residual_block(last_step_output)
        output = self.dec(z)
        return output
    
    def forward_with_latent(self, x_sequence):
        """Forward pass that also returns the latent representation for FiLM."""
        B, S, C, H, W = x_sequence.shape
        x_reshaped = x_sequence.view(B * S, C, H, W)
        z_reshaped = self.enc(x_reshaped)
        _, hidden_C, hidden_H, hidden_W = z_reshaped.shape
        z_sequence = z_reshaped.view(B, S, hidden_C, hidden_H, hidden_W)
        lstm_out_sequence, _ = self.conv_lstm(z_sequence)
        last_step_output = lstm_out_sequence[:, -1, :, :, :]
        z = self.residual_block(last_step_output)
        return z, hidden_C


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) Layer."""
    
    def __init__(self, conditioning_dim, feature_channels):
        super().__init__()
        self.gamma_proj = nn.Linear(conditioning_dim, feature_channels)
        self.beta_proj = nn.Linear(conditioning_dim, feature_channels)
        
        nn.init.ones_(self.gamma_proj.weight.data[:, 0])
        if conditioning_dim > 1:
            nn.init.zeros_(self.gamma_proj.weight.data[:, 1:])
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
    
    def forward(self, features, conditioning):
        gamma = self.gamma_proj(conditioning).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_proj(conditioning).unsqueeze(-1).unsqueeze(-1)
        return gamma * features + beta


class MultiParamSurrogateModel(nn.Module):
    """
    Surrogate model with FiLM conditioning for all 4 parameters (Ra, Ha, Q, Da).
    
    Architecture:
    1. Parameters are normalized and concatenated into a 4-dim vector
    2. Parameter vector is encoded through an MLP to get a conditioning embedding
    3. Input sequence is concatenated with 4 parameter channels and processed by STNNN encoder
    4. FiLM layer modulates the latent features based on parameter embedding
    5. Modulated features are decoded to predict the next timestep
    """
    
    PARAM_RANGES = {
        'Ra': (100.0, 1e7),
        'Ha': (0.0, 50.0),
        'Q': (-6.0, 6.0),
        'Da': (0.001, 0.15),
    }
    
    def __init__(self, model_config, use_film=True):
        super().__init__()
        self.use_film = use_film
        
        original_channels = model_config.get('output_ch', 4)
        # 4 extra channels: one for each parameter (Ra, Ha, Q, Da)
        new_input_channels = original_channels + 4
        
        surrogate_config = model_config.copy()
        surrogate_config['input_ch'] = new_input_channels
        surrogate_config['output_ch'] = original_channels
        
        self.stnnn = SequenceSTNNN(**surrogate_config)
        self.hidden_channels = model_config.get('hidden', 128)
        
        # Multi-parameter encoder: 4 params -> 128-dim embedding
        self.param_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128)
        )
        
        if self.use_film:
            self.film_layer = FiLMLayer(
                conditioning_dim=128,
                feature_channels=self.hidden_channels
            )
            self.film_decoder = nn.Sequential(
                nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(self.hidden_channels, original_channels, 1)
            )

    def normalize_params(self, ra, ha, q, da):
        """
        Normalize all parameters to roughly [0, 1] range.
        Ra uses log-scale due to its wide range.
        """
        # Ra: log-scale normalization
        ra_min, ra_max = self.PARAM_RANGES['Ra']
        log_ra = torch.log10(torch.clamp(ra, min=1.0))
        log_min = np.log10(max(ra_min, 1.0))
        log_max = np.log10(max(ra_max, 1.0))
        ra_norm = (log_ra - log_min) / (log_max - log_min + 1e-8)
        
        # Ha: linear normalization
        ha_min, ha_max = self.PARAM_RANGES['Ha']
        ha_norm = (ha - ha_min) / (ha_max - ha_min + 1e-8)
        
        # Q: linear normalization (can be negative)
        q_min, q_max = self.PARAM_RANGES['Q']
        q_norm = (q - q_min) / (q_max - q_min + 1e-8)
        
        # Da: linear normalization
        da_min, da_max = self.PARAM_RANGES['Da']
        da_norm = (da - da_min) / (da_max - da_min + 1e-8)
        
        return ra_norm, ha_norm, q_norm, da_norm

    def forward(self, x_sequence, ra_scalar, ha_scalar, q_scalar, da_scalar, use_film=None):
        """
        Forward pass.
        
        Args:
            x_sequence: [B, S, 4, H, W] input sequence
            ra_scalar, ha_scalar, q_scalar, da_scalar: [B] parameter tensors
        """
        if use_film is None:
            use_film = self.use_film
            
        B, S, C, H, W = x_sequence.shape
        
        # Normalize parameters
        ra_n, ha_n, q_n, da_n = self.normalize_params(ra_scalar, ha_scalar, q_scalar, da_scalar)
        
        # Create parameter channels [B, S, 4, H, W]
        ra_ch = ra_n.view(B, 1, 1, 1, 1).expand(B, S, 1, H, W)
        ha_ch = ha_n.view(B, 1, 1, 1, 1).expand(B, S, 1, H, W)
        q_ch = q_n.view(B, 1, 1, 1, 1).expand(B, S, 1, H, W)
        da_ch = da_n.view(B, 1, 1, 1, 1).expand(B, S, 1, H, W)
        
        model_input = torch.cat([x_sequence, ra_ch, ha_ch, q_ch, da_ch], dim=2)
        
        if use_film:
            z, _ = self.stnnn.forward_with_latent(model_input)
            # Build conditioning vector
            param_vec = torch.stack([ra_n, ha_n, q_n, da_n], dim=-1)  # [B, 4]
            param_embed = self.param_encoder(param_vec)  # [B, 128]
            z_modulated = self.film_layer(z, param_embed)
            output = self.film_decoder(z_modulated)
        else:
            output = self.stnnn(model_input)
        
        return output


# =============================================================================
# Component 3: Multi-Parameter Sequence Dataset
# =============================================================================
class MultiParamSequenceDataset(Dataset):
    """
    Dataset that provides sequences of frames along with ALL 4 parameter values.
    Each sample: input sequence [S, C, H, W], target frame [C, H, W], param_dict.
    """
    
    def __init__(self, mat_file_path, sequence_length=3, device='cpu'):
        self.mat_dataset = MatDataset(mat_file_path, device=device)
        self.sequence_length = sequence_length
        self.device = device
        
        # Extract all 4 target parameters
        params = self.mat_dataset.params
        self.ra_value = torch.tensor(params.get('Ra', 1e4), dtype=torch.float32, device=device)
        self.ha_value = torch.tensor(params.get('Ha', 0.0), dtype=torch.float32, device=device)
        self.q_value = torch.tensor(params.get('Q', 0.0), dtype=torch.float32, device=device)
        self.da_value = torch.tensor(params.get('Da', 1e-3), dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.mat_dataset) - (self.sequence_length - 1)

    def __getitem__(self, idx):
        input_frames = [self.mat_dataset[idx + i][0] for i in range(self.sequence_length)]
        _, target_frame, _ = self.mat_dataset[idx + self.sequence_length - 1]
        input_sequence = torch.stack(input_frames, dim=0)
        
        # Return all 4 parameter values
        param_dict = {
            'Ra': self.ra_value,
            'Ha': self.ha_value,
            'Q': self.q_value,
            'Da': self.da_value,
        }
        return input_sequence, target_frame, param_dict


def collate_multi_param(batch):
    """Custom collate function to handle param_dict in DataLoader."""
    input_seqs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    
    # Stack parameter values
    param_dict = {
        'Ra': torch.stack([item[2]['Ra'] for item in batch]),
        'Ha': torch.stack([item[2]['Ha'] for item in batch]),
        'Q': torch.stack([item[2]['Q'] for item in batch]),
        'Da': torch.stack([item[2]['Da'] for item in batch]),
    }
    return input_seqs, targets, param_dict


# =============================================================================
# Component 4: Training Function
# =============================================================================
def calculate_normalization_weights(model, dataloader, physics_loss_fn, device, num_samples=5):
    """Calculate normalization weights for physics losses using multiple samples."""
    print("  Calculating normalization weights...")
    model.eval()
    
    accumulated_losses = {'continuity': [], 'momentum_x': [], 'momentum_y': [], 'energy': []}
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (input_seq, _, param_dict) in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
                
            input_seq = input_seq.to(device)
            ra_vals = param_dict['Ra'].to(device)
            ha_vals = param_dict['Ha'].to(device)
            q_vals = param_dict['Q'].to(device)
            da_vals = param_dict['Da'].to(device)
            
            pred_state = model(input_seq, ra_vals, ha_vals, q_vals, da_vals)
            state_t = input_seq[:, -1, :, :, :]
            
            # Process each sample in the batch
            for i in range(input_seq.shape[0]):
                initial_losses = physics_loss_fn(
                    state_t[i:i+1], pred_state[i:i+1],
                    ra_value=ra_vals[i], ha_value=ha_vals[i],
                    q_value=q_vals[i], da_value=da_vals[i],
                    validation_mode=True
                )
                
                for k in ['continuity', 'momentum_x', 'momentum_y', 'energy']:
                    if k in initial_losses:
                        loss_val = initial_losses[k].item()
                        if torch.isfinite(initial_losses[k]) and not (np.isnan(loss_val) or np.isinf(loss_val)):
                            accumulated_losses[k].append(loss_val)
                sample_count += 1
                
                if sample_count >= num_samples:
                    break
            if sample_count >= num_samples:
                break
    
    # Calculate mean losses
    mean_losses = {}
    for k in ['continuity', 'momentum_x', 'momentum_y', 'energy']:
        if accumulated_losses[k]:
            mean_losses[k] = np.mean(accumulated_losses[k])
        else:
            mean_losses[k] = 1.0
            print(f"    Warning: No valid samples for {k}, using default loss 1.0")
    
    # Calculate weights with safety checks
    weights = {}
    for k in ['continuity', 'momentum_x', 'momentum_y', 'energy']:
        loss_val = mean_losses[k]
        
        if loss_val < 1e-10:
            # Loss is too small, use large weight but cap it
            weights[k] = 1e10
            print(f"    Warning: {k} mean loss is very small ({loss_val:.2e}), capping weight to 1e10")
        elif loss_val > 1e10:
            # Loss is too large, use small weight but don't let it be zero
            weights[k] = 1e-10
            print(f"    Warning: {k} mean loss is very large ({loss_val:.2e}), using minimum weight 1e-10")
        else:
            weights[k] = 1.0 / (loss_val + 1e-8)
    
    # Print individual loss values for debugging
    print(f"  Mean losses (from {sample_count} samples): " + 
          ", ".join(f"{k}={mean_losses[k]:.6e}" for k in ['continuity', 'momentum_x', 'momentum_y', 'energy']))
    print(f"  Norm weights: " + ", ".join(f"{k}={v:.4f}" for k, v in weights.items()))
    return weights


def train_model(train_loader, val_loader, model, physics_loss_fn, optimizer, scheduler,
                norm_weights, config):
    """Train the multi-parameter surrogate model."""
    device = next(model.parameters()).device
    best_val_loss = float('inf')
    
    checkpoint_dir = config.get('checkpoint_dir', None)
    checkpoint_interval = config.get('checkpoint_interval', 20)
    
    print(f"  Starting training for {config['epochs']} epochs...")
    if checkpoint_dir:
        print(f"  Checkpoints every {checkpoint_interval} epochs -> {checkpoint_dir}")
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        
        for input_seq, target_frame, param_dict in train_loader:
            input_seq = input_seq.to(device, non_blocking=True)
            target_frame = target_frame.to(device, non_blocking=True)
            ra_vals = param_dict['Ra'].to(device, non_blocking=True)
            ha_vals = param_dict['Ha'].to(device, non_blocking=True)
            q_vals = param_dict['Q'].to(device, non_blocking=True)
            da_vals = param_dict['Da'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            pred_frame = model(input_seq, ra_vals, ha_vals, q_vals, da_vals)
            loss_data = F.mse_loss(pred_frame, target_frame)
            
            state_t = input_seq[:, -1, :, :, :]
            p_losses = physics_loss_fn(
                state_t, pred_frame,
                ra_value=ra_vals[0], ha_value=ha_vals[0],
                q_value=q_vals[0], da_value=da_vals[0],
                validation_mode=True
            )
            
            loss_physics = (
                p_losses['continuity'] * norm_weights['continuity'] +
                p_losses['momentum_x'] * norm_weights['momentum_x'] +
                p_losses['momentum_y'] * norm_weights['momentum_y'] * config.get('y_momentum_weight', 3.0) +
                p_losses['energy'] * norm_weights['energy'] * config.get('energy_weight', 3.0)
            )
            
            total_loss = loss_data + config['physics_lambda'] * loss_physics
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += total_loss.item()
        
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_seq, target_frame, param_dict in val_loader:
                input_seq = input_seq.to(device, non_blocking=True)
                target_frame = target_frame.to(device, non_blocking=True)
                ra_vals = param_dict['Ra'].to(device, non_blocking=True)
                ha_vals = param_dict['Ha'].to(device, non_blocking=True)
                q_vals = param_dict['Q'].to(device, non_blocking=True)
                da_vals = param_dict['Da'].to(device, non_blocking=True)
                
                pred_frame = model(input_seq, ra_vals, ha_vals, q_vals, da_vals)
                loss_data = F.mse_loss(pred_frame, target_frame)
                
                state_t = input_seq[:, -1, :, :, :]
                p_losses = physics_loss_fn(
                    state_t, pred_frame,
                    ra_value=ra_vals[0], ha_value=ha_vals[0],
                    q_value=q_vals[0], da_value=da_vals[0],
                    validation_mode=True
                )
                loss_physics = sum(p_losses[k] * norm_weights[k] for k in norm_weights if k in p_losses)
                
                total_loss = loss_data + config['physics_lambda'] * loss_physics
                val_loss += total_loss.item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch [{epoch+1:3d}/{config['epochs']}], "
                  f"Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")
        
        # Periodic checkpoints
        if checkpoint_dir and (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1:04d}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
            }, ckpt_path)
            print(f"    Checkpoint saved: {ckpt_path}")
        
        # Best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config['model_save_path'])
            if not (checkpoint_dir and (epoch + 1) % checkpoint_interval == 0):
                pass  # Avoid duplicate message

    print(f"  Training finished. Best Val Loss: {best_val_loss:.6f}")


# =============================================================================
# Component 5: Multi-Parameter Inference
# =============================================================================
def predict_multi_params(model, physics_loss_fn, input_sequence, target_frame, config, device,
                         norm_weights=None, stable_frame_data=None):
    """
    Simultaneously infer Ra, Ha, Q, Da using physics-based optimization.
    
    The inference process optimizes all 4 parameters to minimize:
    1. Data Loss: MSE between prediction and target
    2. Physics Residual Loss: All 4 governing equations
    3. Consistency Losses: Agreement between inferred and guessed values for each param
    
    Args:
        model: Trained surrogate model (frozen)
        physics_loss_fn: Physics loss calculator
        input_sequence: Input sequence [S, C, H, W]
        target_frame: Target frame [C, H, W]
        config: Configuration dict
        device: Computation device
        norm_weights: Normalization weights for physics losses
        stable_frame_data: If provided, use this stable frame data directly [C, H, W]
        
    Returns:
        best_params: dict with best Ra, Ha, Q, Da values
        history: dict of optimization histories for each param
    """
    model.eval()
    
    if norm_weights is None:
        norm_weights = {'continuity': 1.0, 'momentum_x': 1.0, 'momentum_y': 1.0, 'energy': 1.0}
    
    # Initialize parameter guesses
    ra_guess = torch.tensor([config['ra_initial_guess']], device=device, dtype=torch.float32, requires_grad=True)
    ha_guess = torch.tensor([config['ha_initial_guess']], device=device, dtype=torch.float32, requires_grad=True)
    q_guess = torch.tensor([config['q_initial_guess']], device=device, dtype=torch.float32, requires_grad=True)
    da_guess = torch.tensor([config['da_initial_guess']], device=device, dtype=torch.float32, requires_grad=True)
    
    optimizer = optim.Adam([ra_guess, ha_guess, q_guess, da_guess], lr=config['inference_lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['inference_steps'], eta_min=config['inference_lr'] * 0.01
    )
    
    history = {'Ra': [], 'Ha': [], 'Q': [], 'Da': [], 'loss': []}
    best_params = {
        'Ra': ra_guess.item(), 'Ha': ha_guess.item(),
        'Q': q_guess.item(), 'Da': da_guess.item()
    }
    best_loss = float('inf')
    
    input_seq_batch = input_sequence.unsqueeze(0)
    target_batch = target_frame.unsqueeze(0)
    
    # Stable frame setup
    if stable_frame_data is not None:
        stable_frame_batch = stable_frame_data.unsqueeze(0).to(device)
        state_t_stable = stable_frame_batch
        pred_frame_stable = stable_frame_batch
    
    for step in range(config['inference_steps']):
        optimizer.zero_grad()
        
        if stable_frame_data is not None:
            pred_frame = pred_frame_stable
            state_t = state_t_stable
            loss_data = torch.tensor(0.0, device=device)
        else:
            pred_frame = model(input_seq_batch, ra_guess, ha_guess, q_guess, da_guess)
            loss_data = F.mse_loss(pred_frame, target_batch)
            state_t = input_seq_batch[:, -1, :, :, :]
        
        is_steady = (stable_frame_data is not None)
        
        # Physics residual loss (all equations, all params)
        physics_losses = physics_loss_fn.physics_residual_loss(
            state_t, pred_frame,
            ra_value=ra_guess, ha_value=ha_guess,
            q_value=q_guess, da_value=da_guess,
            steady_state=is_steady
        )
        
        loss_physics = (
            physics_losses['continuity'] * norm_weights.get('continuity', 1.0) +
            physics_losses['momentum_x'] * norm_weights.get('momentum_x', 1.0) +
            physics_losses['momentum_y'] * norm_weights.get('momentum_y', 1.0) * config.get('y_momentum_weight', 3.0) +
            physics_losses['energy'] * norm_weights.get('energy', 1.0) * config.get('energy_weight', 3.0)
        )
        
        # Consistency losses for each parameter
        U_now, V_now, T_now, P_now = torch.chunk(state_t, 4, 1)
        U_next, V_next, T_next, P_next = torch.chunk(pred_frame, 4, 1)
        
        loss_da_cons, da_inferred = physics_loss_fn.da_consistency_loss(
            U_now, V_now, P_next, U_next, V_next, da_guess, steady_state=is_steady
        )
        loss_ra_cons, ra_inferred = physics_loss_fn.ra_consistency_loss(
            U_now, V_now, P_next, U_next, V_next, T_next, ra_guess,
            da_value=da_guess, ha_value=ha_guess, steady_state=is_steady
        )
        loss_ha_cons, ha_inferred = physics_loss_fn.ha_consistency_loss(
            U_now, V_now, P_next, U_next, V_next, T_next, ha_guess,
            ra_value=ra_guess, da_value=da_guess, steady_state=is_steady
        )
        loss_q_cons, q_inferred = physics_loss_fn.q_consistency_loss(
            U_now, V_now, T_now, T_next, q_guess, steady_state=is_steady
        )
        
        # Total loss
        total_loss = (
            config.get('data_weight', 1.0) * loss_data +
            config.get('physics_lambda', 2.0) * loss_physics +
            config.get('da_consistency_weight', 2.0) * loss_da_cons +
            config.get('ra_consistency_weight', 2.0) * loss_ra_cons +
            config.get('ha_consistency_weight', 2.0) * loss_ha_cons +
            config.get('q_consistency_weight', 2.0) * loss_q_cons
        )
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Clamp parameters to valid ranges
        with torch.no_grad():
            ra_guess.data.clamp_(min=config['ra_min'], max=config['ra_max'])
            ha_guess.data.clamp_(min=config['ha_min'], max=config['ha_max'])
            q_guess.data.clamp_(min=config['q_min'], max=config['q_max'])
            da_guess.data.clamp_(min=config['da_min'], max=config['da_max'])
        
        # Record history
        current = {
            'Ra': ra_guess.item(), 'Ha': ha_guess.item(),
            'Q': q_guess.item(), 'Da': da_guess.item()
        }
        for k in current:
            history[k].append(current[k])
        history['loss'].append(total_loss.item())
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_params = current.copy()
        
        if (step + 1) % 200 == 0:
            print(f"    Step {step+1}: "
                  f"Ra={current['Ra']:.2f}, Ha={current['Ha']:.2f}, "
                  f"Q={current['Q']:.3f}, Da={current['Da']:.5f}, "
                  f"Loss={total_loss.item():.6f}")
            print(f"      Inferred: Ra={ra_inferred.item():.2f}, Ha={ha_inferred.item():.2f}, "
                  f"Q={q_inferred.item():.3f}, Da={da_inferred.item():.5f}")
    
    return best_params, history


def predict_multi_params_multi_timestep(model, physics_loss_fn, dataset, config, device,
                                        norm_weights=None, steady_state=False):
    """
    Infer all 4 parameters using multiple timesteps for more stable estimation.
    """
    model.eval()
    
    if norm_weights is None:
        norm_weights = {'continuity': 1.0, 'momentum_x': 1.0, 'momentum_y': 1.0, 'energy': 1.0}
    
    num_samples = min(config.get('num_inference_samples', 10), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    
    ra_guess = torch.tensor([config['ra_initial_guess']], device=device, dtype=torch.float32, requires_grad=True)
    ha_guess = torch.tensor([config['ha_initial_guess']], device=device, dtype=torch.float32, requires_grad=True)
    q_guess = torch.tensor([config['q_initial_guess']], device=device, dtype=torch.float32, requires_grad=True)
    da_guess = torch.tensor([config['da_initial_guess']], device=device, dtype=torch.float32, requires_grad=True)
    
    optimizer = optim.Adam([ra_guess, ha_guess, q_guess, da_guess], lr=config['inference_lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['inference_steps'], eta_min=config['inference_lr'] * 0.01
    )
    
    history = {'Ra': [], 'Ha': [], 'Q': [], 'Da': [], 'loss': []}
    best_params = {
        'Ra': ra_guess.item(), 'Ha': ha_guess.item(),
        'Q': q_guess.item(), 'Da': da_guess.item()
    }
    best_loss = float('inf')
    
    for step in range(config['inference_steps']):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        
        for idx in indices:
            input_seq, target_frame, param_dict_sample = dataset[idx]
            input_seq_batch = input_seq.unsqueeze(0).to(device)
            target_batch = target_frame.unsqueeze(0).to(device)
            
            pred_frame = model(input_seq_batch, ra_guess, ha_guess, q_guess, da_guess)
            loss_data = F.mse_loss(pred_frame, target_batch)
            
            state_t = input_seq_batch[:, -1, :, :, :]
            
            physics_losses = physics_loss_fn.physics_residual_loss(
                state_t, pred_frame,
                ra_value=ra_guess, ha_value=ha_guess,
                q_value=q_guess, da_value=da_guess,
                steady_state=steady_state
            )
            
            loss_physics = (
                physics_losses['continuity'] * norm_weights.get('continuity', 1.0) +
                physics_losses['momentum_x'] * norm_weights.get('momentum_x', 1.0) +
                physics_losses['momentum_y'] * norm_weights.get('momentum_y', 1.0) * config.get('y_momentum_weight', 3.0) +
                physics_losses['energy'] * norm_weights.get('energy', 1.0) * config.get('energy_weight', 3.0)
            )
            
            # Consistency losses
            U_now, V_now, T_now, P_now = torch.chunk(state_t, 4, 1)
            U_next, V_next, T_next, P_next = torch.chunk(pred_frame, 4, 1)
            
            loss_da_cons, _ = physics_loss_fn.da_consistency_loss(
                U_now, V_now, P_next, U_next, V_next, da_guess, steady_state=steady_state
            )
            loss_ra_cons, _ = physics_loss_fn.ra_consistency_loss(
                U_now, V_now, P_next, U_next, V_next, T_next, ra_guess,
                da_value=da_guess, ha_value=ha_guess, steady_state=steady_state
            )
            loss_ha_cons, _ = physics_loss_fn.ha_consistency_loss(
                U_now, V_now, P_next, U_next, V_next, T_next, ha_guess,
                ra_value=ra_guess, da_value=da_guess, steady_state=steady_state
            )
            loss_q_cons, _ = physics_loss_fn.q_consistency_loss(
                U_now, V_now, T_now, T_next, q_guess, steady_state=steady_state
            )
            
            sample_loss = (
                config.get('data_weight', 1.0) * loss_data +
                config.get('physics_lambda', 2.0) * loss_physics +
                config.get('da_consistency_weight', 2.0) * loss_da_cons +
                config.get('ra_consistency_weight', 2.0) * loss_ra_cons +
                config.get('ha_consistency_weight', 2.0) * loss_ha_cons +
                config.get('q_consistency_weight', 2.0) * loss_q_cons
            )
            total_loss = total_loss + sample_loss
        
        total_loss = total_loss / num_samples
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        with torch.no_grad():
            ra_guess.data.clamp_(min=config['ra_min'], max=config['ra_max'])
            ha_guess.data.clamp_(min=config['ha_min'], max=config['ha_max'])
            q_guess.data.clamp_(min=config['q_min'], max=config['q_max'])
            da_guess.data.clamp_(min=config['da_min'], max=config['da_max'])
        
        current = {
            'Ra': ra_guess.item(), 'Ha': ha_guess.item(),
            'Q': q_guess.item(), 'Da': da_guess.item()
        }
        for k in current:
            history[k].append(current[k])
        history['loss'].append(total_loss.item())
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_params = current.copy()
        
        if (step + 1) % 200 == 0:
            print(f"    Step {step+1}: "
                  f"Ra={current['Ra']:.2f}, Ha={current['Ha']:.2f}, "
                  f"Q={current['Q']:.3f}, Da={current['Da']:.5f}, "
                  f"Loss={total_loss.item():.6f}")
    
    return best_params, history


# =============================================================================
# Component 6: Visualization
# =============================================================================
def plot_multi_param_convergence(history, true_values, file_name, save_dir):
    """Plot convergence curves for all 4 parameters."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    params = ['Ra', 'Ha', 'Q', 'Da']
    colors = ['blue', 'green', 'orange', 'purple']
    
    for ax, param, color in zip(axes.flat, params, colors):
        steps = range(len(history[param]))
        ax.plot(steps, history[param], label=f'Predicted {param}', color=color, linewidth=2)
        
        true_val = true_values.get(param)
        if true_val is not None:
            ax.axhline(y=true_val, color='r', linestyle='--', linewidth=2,
                       label=f'True {param} ({true_val:.4f})')
        
        ax.set_xlabel('Optimization Steps', fontsize=11)
        ax.set_ylabel(param, fontsize=11)
        ax.set_title(f'{param} Convergence', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.5)
    
    fig.suptitle(f'Multi-Parameter Convergence: {file_name}', fontsize=15)
    plt.tight_layout()
    
    base_name = os.path.splitext(file_name)[0]
    save_path = os.path.join(save_dir, f'multi_convergence_{base_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Convergence plot saved: {save_path}")


def plot_multi_param_accuracy(results, save_dir):
    """Plot prediction accuracy scatter plots for all 4 parameters."""
    params = ['Ra', 'Ha', 'Q', 'Da']
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    colors = ['blue', 'green', 'orange', 'purple']
    
    for ax, param, color in zip(axes.flat, params, colors):
        true_vals = []
        pred_vals = []
        
        for r in results:
            tv = r.get('true_values', {}).get(param)
            pv = r.get('predicted_values', {}).get(param)
            if tv is not None and pv is not None:
                true_vals.append(tv)
                pred_vals.append(pv)
        
        if not true_vals:
            ax.text(0.5, 0.5, f'No data for {param}', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            continue
        
        ax.scatter(true_vals, pred_vals, alpha=0.7, c=color, edgecolors='k', s=120)
        
        min_val = min(min(true_vals), min(pred_vals))
        max_val = max(max(true_vals), max(pred_vals))
        margin = (max_val - min_val) * 0.1 if max_val != min_val else abs(max_val) * 0.1 + 0.1
        
        ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin],
                'r--', linewidth=2, label='Perfect Fit')
        
        ax.set_xlabel(f'True {param}', fontsize=12)
        ax.set_ylabel(f'Predicted {param}', fontsize=12)
        ax.set_title(f'{param}: True vs Predicted', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # MAPE
        errors = [abs(t - p) / (abs(t) + 1e-8) * 100 for t, p in zip(true_vals, pred_vals)]
        mape = np.mean(errors) if errors else 0
        ax.text(0.05, 0.95, f'MAPE: {mape:.2f}%', transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat'))
    
    fig.suptitle('Multi-Parameter Prediction Accuracy', fontsize=16)
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, 'multi_param_accuracy.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Prediction accuracy plot saved: {plot_path}")


# =============================================================================
# Component 7: Data Discovery
# =============================================================================
def find_all_param_files(data_root, base_fluid):
    """
    Find ALL .mat files across ALL parameter subfolders (Da, Ha, Q, Ra)
    under the given base fluid directory.
    
    Excludes Phi subfolders as per user request.
    
    The function identifies the varying parameter from the filename itself
    (e.g., '01_Da_0.100.mat' -> Da, '01_Ha_0.mat' -> Ha).
    
    Returns:
        list of (file_path, varying_param_name) tuples
    """
    import re
    
    base_path = os.path.join(data_root, base_fluid)
    
    if not os.path.exists(base_path):
        # Case-insensitive fallback
        if os.path.exists(data_root):
            dirs = os.listdir(data_root)
            candidates = [d for d in dirs if base_fluid.lower() == d.lower()]
            if candidates:
                base_path = os.path.join(data_root, candidates[0])
            else:
                print(f"  Error: Base path {base_path} not found.")
                return []
    
    print(f"  Searching in {base_path} for all parameter folders...")
    
    # Find all .mat files recursively
    all_mat_files = glob.glob(os.path.join(base_path, "**", "*.mat"), recursive=True)
    
    # Target parameters (excluding Phi)
    target_params = ['Da', 'Ha', 'Q', 'Ra']
    
    # Regex pattern to match parameter in filename: _Da_, _Ha_, _Q_, _Ra_
    # Handles: 01_Da_0.100.mat, 01_Ha_0.mat, 01_Q_-6.0.mat, 01_Ra_1000.mat
    param_pattern = re.compile(r'_(' + '|'.join(target_params) + r')_', re.IGNORECASE)
    # Also handle "Darcy" as Da
    darcy_pattern = re.compile(r'[Dd]arcy', re.IGNORECASE)
    
    categorized = []
    skipped_phi = 0
    skipped_unknown = 0
    
    for f in sorted(all_mat_files):
        file_name = os.path.basename(f)
        full_path_lower = f.lower()
        
        # Skip Phi files (check both filename and full path)
        if 'phi' in full_path_lower:
            skipped_phi += 1
            continue
        
        # Try to identify parameter from filename
        match = param_pattern.search(file_name)
        if match:
            param = match.group(1)
            # Normalize case (e.g., 'da' -> 'Da')
            param_normalized = next((p for p in target_params if p.lower() == param.lower()), None)
            if param_normalized:
                categorized.append((f, param_normalized))
                continue
        
        # Check for "Darcy" in path (some folders use "Darcy" instead of "Da")
        if darcy_pattern.search(f):
            # Verify the file itself contains Da-like data
            if '_Da_' in file_name or darcy_pattern.search(file_name):
                categorized.append((f, 'Da'))
                continue
            # Check parent directories
            for part in f.replace('\\', '/').split('/'):
                if darcy_pattern.search(part):
                    categorized.append((f, 'Da'))
                    break
            else:
                skipped_unknown += 1
                continue
            continue
        
        # Fallback: check parent directory names
        path_parts = f.replace('\\', '/').split('/')
        found = False
        for part in reversed(path_parts[:-1]):  # Skip filename itself
            for param in target_params:
                if f"_{param}" in part or part.endswith(f"_{param}"):
                    categorized.append((f, param))
                    found = True
                    break
            if found:
                break
        
        if not found:
            skipped_unknown += 1
    
    if skipped_phi > 0:
        print(f"  Skipped {skipped_phi} Phi files (excluded)")
    if skipped_unknown > 0:
        print(f"  Skipped {skipped_unknown} unrecognized files")
    
    return categorized


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Multi-Parameter Inverse Problem Solver (Ra, Ha, Q, Da)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training and inference
  python train_and_infer_multi.py --base_fluid EG
  
  # Skip training, use existing model
  python train_and_infer_multi.py --base_fluid EG --skip_train
  
  # Multi-timestep inference
  python train_and_infer_multi.py --base_fluid EG --skip_train --multi_timestep
  
  # Enable FiLM conditioning
  python train_and_infer_multi.py --base_fluid EG --use_film
  
  # Adjust training epochs
  python train_and_infer_multi.py --base_fluid EG --epochs 200
  
  # Use stable frame for inference
  python train_and_infer_multi.py --base_fluid EG --skip_train --stable_frame_idx 100
        """
    )
    parser.add_argument('--base_fluid', type=str, default='EG', help='Base fluid name (e.g., EG, Water)')
    parser.add_argument('--data_root', type=str, default='data', help='Root data directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--results_dir', type=str, default='multi_inference_results', help='Output directory')
    parser.add_argument('--skip_train', action='store_true', help='Skip training phase')
    parser.add_argument('--inference_lr', type=float, default=0.005, help='Learning rate for inference')
    parser.add_argument('--inference_steps', type=int, default=1500, help='Optimization steps for inference')
    parser.add_argument('--multi_timestep', action='store_true', help='Use multi-timestep inference')
    parser.add_argument('--use_film', action='store_true', help='Enable FiLM conditioning')
    parser.add_argument('--test_epochs', type=int, default=None,
                        help='Override inference steps for test')
    parser.add_argument('--stable_frame_idx', type=int, default=None,
                        help='Use specific stable frame index for inference')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--checkpoint_interval', type=int, default=20,
                        help='Save checkpoint every N epochs')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"FiLM conditioning: {'Enabled' if args.use_film else 'Disabled'}")
    
    # =====================================================================
    # 1. Data Preparation
    # =====================================================================
    print("\n" + "="*80)
    print("DATA PREPARATION")
    print("="*80)
    
    categorized_files = find_all_param_files(args.data_root, args.base_fluid)
    
    if not categorized_files:
        raise ValueError(f"No .mat files found for {args.base_fluid}")
    
    # Count per parameter
    param_counts = {}
    for _, param in categorized_files:
        param_counts[param] = param_counts.get(param, 0) + 1
    
    print(f"  Found {len(categorized_files)} total files:")
    for p, c in sorted(param_counts.items()):
        print(f"    {p}: {c} files")
    
    # Shuffle and split: 80% train, 10% val, 10% test
    all_files = [f for f, _ in categorized_files]
    all_params = [p for _, p in categorized_files]
    
    # Use reproducible shuffle
    combined = list(zip(all_files, all_params))
    random.seed(42)
    random.shuffle(combined)
    
    n_total = len(combined)
    n_test = max(4, int(n_total * 0.1))   # At least 4 test files (1 per param ideally)
    n_val = max(4, int(n_total * 0.1))     # At least 4 val files
    n_train = n_total - n_test - n_val
    
    test_combined = combined[:n_test]
    val_combined = combined[n_test:n_test + n_val]
    train_combined = combined[n_test + n_val:]
    
    train_files = [f for f, _ in train_combined]
    val_files = [f for f, _ in val_combined]
    test_files = [f for f, _ in test_combined]
    test_params = [p for _, p in test_combined]
    
    print(f"\n  Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    # Show test file details
    print(f"\n  Test files:")
    for f, p in test_combined:
        print(f"    [{p}] {os.path.basename(f)}")
    
    # Setup save directory
    run_name = f"{args.base_fluid}_multi"
    save_dir = os.path.join(args.results_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    
    # Auto-skip training if checkpoint exists
    if os.path.exists(checkpoint_path) and not args.skip_train:
        print(f"\n[INFO] Found existing checkpoint. Skipping training.")
        args.skip_train = True
    
    # =====================================================================
    # Configuration
    # =====================================================================
    inference_steps = args.test_epochs if args.test_epochs is not None else args.inference_steps
    
    config = {
        'epochs': args.epochs,
        'learning_rate': 0.0005,
        'physics_lambda': 2.0,
        'y_momentum_weight': 3.0,
        'energy_weight': 3.0,
        'scheduler_patience': 15,
        'scheduler_factor': 0.5,
        'batch_size': args.batch_size,
        'sequence_length': 3,
        'model_save_path': checkpoint_path,
        'checkpoint_dir': save_dir,
        'checkpoint_interval': args.checkpoint_interval,
        # Inference settings
        'inference_lr': args.inference_lr,
        'inference_steps': inference_steps,
        'data_weight': 1.0,
        'num_inference_samples': 10,
        # Ra
        'ra_initial_guess': 1e5,
        'ra_min': 100.0,
        'ra_max': 1e8,
        'ra_consistency_weight': 2.0,
        # Ha
        'ha_initial_guess': 25.0,
        'ha_min': 0.0,
        'ha_max': 100.0,
        'ha_consistency_weight': 2.0,
        # Q
        'q_initial_guess': 0.0,
        'q_min': -10.0,
        'q_max': 10.0,
        'q_consistency_weight': 2.0,
        # Da
        'da_initial_guess': 0.05,
        'da_min': 0.001,
        'da_max': 0.15,
        'da_consistency_weight': 2.0,
    }
    
    # Save config
    config_save = {k: v for k, v in config.items() if not isinstance(v, torch.Tensor)}
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config_save, f, indent=4, default=str)
    
    # =====================================================================
    # Model
    # =====================================================================
    model_config = {
        'input_ch': 4,
        'output_ch': 4,
        'hidden': 128,
        'upscale': 1,
        'dropout_rate': 0.1
    }
    model = MultiParamSurrogateModel(
        model_config,
        use_film=args.use_film
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {total_params:,}")
    
    # =====================================================================
    # 2. Training Phase
    # =====================================================================
    if not args.skip_train:
        print("\n" + "="*80)
        print("PHASE 1: TRAINING")
        print("="*80)
        print(f"  Training epochs: {args.epochs}")
        print(f"  Batch size: {config['batch_size']}")
        
        # Load datasets on CPU
        print(f"  Loading {len(train_files)} training files...")
        train_datasets = [
            MultiParamSequenceDataset(f, sequence_length=config['sequence_length'], device='cpu')
            for f in train_files
        ]
        
        print(f"  Loading {len(val_files)} validation files...")
        val_datasets = [
            MultiParamSequenceDataset(f, sequence_length=config['sequence_length'], device='cpu')
            for f in val_files
        ]
        
        num_workers = 0  # Windows
        train_loader = DataLoader(
            ConcatDataset(train_datasets),
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=None if num_workers == 0 else 2,
            collate_fn=collate_multi_param
        )
        val_loader = DataLoader(
            ConcatDataset(val_datasets),
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=None if num_workers == 0 else 2,
            collate_fn=collate_multi_param
        )
        
        print(f"  Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
        
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min',
            patience=config['scheduler_patience'],
            factor=config['scheduler_factor']
        )
        
        # Initialize physics loss from a sample file
        sample_dataset = MatDataset(train_files[0], device=device)
        physics_loss_fn = MultiParamPhysicsLoss(
            params=sample_dataset.get_params(),
            nanofluid_props=sample_dataset.get_nanofluid_properties(),
            dt=sample_dataset.params.get('dt', 0.0001)
        ).to(device)
        
        norm_weights = calculate_normalization_weights(model, train_loader, physics_loss_fn, device)
        train_model(train_loader, val_loader, model, physics_loss_fn, optimizer, scheduler,
                     norm_weights, config)
    else:
        print("\nSkipping training phase.")

    # =====================================================================
    # 3. Inference Phase
    # =====================================================================
    print("\n" + "="*80)
    print("PHASE 2: MULTI-PARAMETER INFERENCE")
    print("="*80)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"  Loading checkpoint from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    inference_results = []
    
    for test_idx, (test_file, varying_param) in enumerate(zip(test_files, test_params)):
        file_name = os.path.basename(test_file)
        print(f"\n  [{test_idx+1}/{len(test_files)}] Processing: {file_name} (varying: {varying_param})")
        
        try:
            ds = MatDataset(test_file, device=device)
            
            true_values = {
                'Ra': ds.params.get('Ra', None),
                'Ha': ds.params.get('Ha', None),
                'Q': ds.params.get('Q', None),
                'Da': ds.params.get('Da', None),
            }
            
            # Initialize physics loss for this file
            physics_loss_fn = MultiParamPhysicsLoss(
                params=ds.get_params(),
                nanofluid_props=ds.get_nanofluid_properties(),
                dt=ds.params.get('dt', 0.0001)
            ).to(device)
            
            # Create sequence dataset
            seq_dataset = MultiParamSequenceDataset(
                test_file, sequence_length=config['sequence_length'], device=device
            )
            
            # Calculate normalization weights for this test file
            test_loader = DataLoader(seq_dataset, batch_size=1, shuffle=False, collate_fn=collate_multi_param)
            test_norm_weights = calculate_normalization_weights(model, test_loader, physics_loss_fn, device)
            
            if args.multi_timestep:
                pred_values, history = predict_multi_params_multi_timestep(
                    model, physics_loss_fn, seq_dataset, config, device,
                    norm_weights=test_norm_weights
                )
            else:
                if args.stable_frame_idx is not None:
                    stable_idx = args.stable_frame_idx
                    if stable_idx >= len(ds):
                        print(f"    Warning: stable_frame_idx {stable_idx} >= dataset length {len(ds)}. Using last frame.")
                        stable_idx = len(ds) - 1
                    
                    stable_frame_data, _, _ = ds[stable_idx]
                    dummy_input_seq = torch.stack([stable_frame_data] * config['sequence_length'], dim=0)
                    dummy_target = stable_frame_data
                    
                    print(f"    Using stable frame at index {stable_idx}")
                    pred_values, history = predict_multi_params(
                        model, physics_loss_fn, dummy_input_seq, dummy_target, config, device,
                        norm_weights=test_norm_weights,
                        stable_frame_data=stable_frame_data
                    )
                else:
                    mid_idx = len(seq_dataset) // 2
                    input_seq, target_frame, _ = seq_dataset[mid_idx]
                    pred_values, history = predict_multi_params(
                        model, physics_loss_fn, input_seq, target_frame, config, device,
                        norm_weights=test_norm_weights,
                        stable_frame_data=None
                    )
            
            # Plot convergence
            plot_multi_param_convergence(history, true_values, file_name, save_dir)
            
            # Build result
            result = {
                'file': file_name,
                'varying_param': varying_param,
                'true_values': true_values,
                'predicted_values': pred_values,
                'percent_errors': {},
            }
            
            print(f"    Results:")
            for param in ['Ra', 'Ha', 'Q', 'Da']:
                tv = true_values.get(param)
                pv = pred_values.get(param)
                if tv is not None and pv is not None:
                    err = abs(tv - pv) / (abs(tv) + 1e-8) * 100
                    result['percent_errors'][param] = err
                    marker = " <-- varying" if param == varying_param else ""
                    print(f"      {param}: True={tv:.4f}, Pred={pv:.4f}, Error={err:.2f}%{marker}")
            
            inference_results.append(result)
            
        except Exception as e:
            print(f"    Error: {e}")
            traceback.print_exc()
    
    # Save results
    results_json = os.path.join(save_dir, 'multi_inference_results.json')
    
    # Convert for JSON serialization
    serializable_results = []
    for r in inference_results:
        sr = {
            'file': r['file'],
            'varying_param': r['varying_param'],
            'true_values': {k: float(v) if v is not None else None for k, v in r['true_values'].items()},
            'predicted_values': {k: float(v) for k, v in r['predicted_values'].items()},
            'percent_errors': {k: float(v) for k, v in r['percent_errors'].items()},
        }
        serializable_results.append(sr)
    
    with open(results_json, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    print(f"\nResults saved to {results_json}")
    
    # =====================================================================
    # 4. Visualization
    # =====================================================================
    print("\n" + "="*80)
    print("PHASE 3: VISUALIZATION")
    print("="*80)
    
    plot_multi_param_accuracy(serializable_results, save_dir)
    
    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Per-parameter MAPE
    param_errors = {'Ra': [], 'Ha': [], 'Q': [], 'Da': []}
    for r in serializable_results:
        for param in ['Ra', 'Ha', 'Q', 'Da']:
            err = r['percent_errors'].get(param)
            if err is not None:
                param_errors[param].append(err)
    
    print(f"\n  Overall MAPE by Parameter:")
    for param in ['Ra', 'Ha', 'Q', 'Da']:
        errs = param_errors[param]
        if errs:
            print(f"    {param}: {np.mean(errs):.2f}% (n={len(errs)})")
        else:
            print(f"    {param}: No data")
    
    print(f"\n  Individual Results:")
    for r in serializable_results:
        print(f"    {r['file']} (varying: {r['varying_param']}):")
        for param in ['Ra', 'Ha', 'Q', 'Da']:
            tv = r['true_values'].get(param)
            pv = r['predicted_values'].get(param)
            err = r['percent_errors'].get(param, 'N/A')
            if tv is not None and pv is not None:
                marker = " *" if param == r['varying_param'] else ""
                print(f"      {param}: True={tv:.4f}, Pred={pv:.4f}, Error={err:.2f}%{marker}")
    
    send_discord_success("Multi-parameter inference completed.", serializable_results)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        error_trace = traceback.format_exc()
        
        print("\n" + "="*80)
        print("ERROR OCCURRED")
        print("="*80)
        print(error_trace)
        
        send_discord_error(error_trace)
        raise

