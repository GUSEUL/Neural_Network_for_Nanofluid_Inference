import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import os

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

def load_mat_file(path):
    """Load MATLAB file supporting both v7.0 and v7.3 formats."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    
    try:
        # Try standard scipy loader first
        return sio.loadmat(path)
    except NotImplementedError:
        # Fallback to h5py for v7.3 files
        if not HAS_H5PY:
            raise ImportError("h5py is required to load MATLAB v7.3 files. Please run: pip install h5py")
        
        with h5py.File(path, 'r') as f:
            # Convert HDF5 structure to dictionary, ignoring metadata
            return {k: np.array(f[k]) for k in f.keys() if not k.startswith('#')}

def extract_nanofluid_properties(d):
    """Extract and calculate nanofluid property ratios with safe scalar extraction."""
    def s(v, df):
        try:
            if v is None: return df
            if hasattr(v, 'item'): return float(v.item()) if v.size == 1 else float(v.flat[0])
            return float(v)
        except: return df
    
    p = {
        'nu_thnf': s(d.get('nuthnf'), 1.0), 'nu_f': s(d.get('nuf'), 1.0),
        'sigma_thnf': s(d.get('sigthnf'), 1.0), 'sigma_f': s(d.get('sigf'), 1.0),
        'rho_thnf': s(d.get('rothnf'), 1000.0), 'rho_f': s(d.get('rof'), 1000.0),
        'beta_thnf': s(d.get('bethnf'), 1.0), 'beta_f': s(d.get('bef'), 1.0),
        'alpha_thnf': s(d.get('althnf'), 1.0), 'alpha_f': s(d.get('alf'), 1.0),
        'rhocp_thnf': s(d.get('rocpthnf'), 1.0), 'rhocp_f': s(d.get('rocpf'), 1.0),
    }
    
    # Calculate key ratios for physics equations
    r = {
        'nu_thnf_ratio': p['nu_thnf'] / p['nu_f'],
        'sigma_thnf_ratio': p['sigma_thnf'] / p['sigma_f'],
        'rho_f_thnf_ratio': p['rho_f'] / p['rho_thnf'],
        'beta_thnf_ratio': p['beta_thnf'] / p['beta_f'],
        'alpha_thnf_ratio': p['alpha_thnf'] / p['alpha_f'],
        'rhocp_f_thnf_ratio': p['rhocp_f'] / p['rhocp_thnf'],
    }
    return {**p, **r}

class MatDataset(Dataset):
    """Dataset class for loading staggered MAC grid data from .mat files."""
    def __init__(self, matfile, device='cpu'):
        # Use the universal loader instead of sio.loadmat directly
        d = load_mat_file(matfile)
        
        self.nanofluid_props = extract_nanofluid_properties(d)
        
        # Stored data arrays: u, v, p, t
        u_raw, v_raw, p_raw, t_raw = d['ustore'], d['vstore'], d['pstore'], d['tstore']
        
        def s(k, df):
            try: return float(d[k].squeeze()) if k in d else df
            except: return df

        self.params = {
            'Ra': s('Ra', 1e4), 'Ha': s('Ha', 0.0), 'Pr': s('Pr', 0.71),
            'Da': s('Da', 1e-3), 'Q': s('Q', 0.0), 'dt': s('dt', 0.0001),
        }

        # Handle dimension ordering: Transpose from (Time, Y, X) to (Y, X, Time) if necessary
        # We assume original shape is (Time, Ny, Nx) or similar based on solver output
        if u_raw.ndim == 3 and u_raw.shape[0] > u_raw.shape[1]: # Likely (Time, Y, X)
            u_raw, v_raw, p_raw, t_raw = [np.transpose(x, (1, 2, 0)) for x in [u_raw, v_raw, p_raw, t_raw]]
        
        # Normalization statistics
        self.u_mu, self.u_std = np.mean(u_raw), np.std(u_raw)
        self.v_mu, self.v_std = np.mean(v_raw), np.std(v_raw)
        self.p_mu, self.p_std = np.mean(p_raw), np.std(p_raw)
        self.t_mu, self.t_std = np.mean(t_raw), np.std(t_raw)
        
        # Convert to tensors
        self.u = torch.tensor((u_raw - self.u_mu) / (self.u_std + 1e-8), dtype=torch.float32, device=device)
        self.v = torch.tensor((v_raw - self.v_mu) / (self.v_std + 1e-8), dtype=torch.float32, device=device)
        self.p = torch.tensor((p_raw - self.p_mu) / (self.p_std + 1e-8), dtype=torch.float32, device=device)
        self.t = torch.tensor((t_raw - self.t_mu) / (self.t_std + 1e-8), dtype=torch.float32, device=device)
        
        self.norm_params = {
            'u': (self.u_mu, self.u_std), 'v': (self.v_mu, self.v_std),
            'p': (self.p_mu, self.p_std), 't': (self.t_mu, self.t_std)
        }
        
        # Grid dimensions: p.shape = (Ny, Nx, Nt)
        self.ny, self.nx, self.nt = self.p.shape
        self.T = self.nt - 1

    def __len__(self):
        return self.T

    def __getitem__(self, idx):
        """Interpolate staggered fields to cell centers and return state pairs."""
        def center(f, axis):
            if axis == 'u': # Staggered in X
                pad = F.pad(f.unsqueeze(0).unsqueeze(0), (1, 1), mode='constant', value=0).squeeze()
                return 0.5 * (pad[:, :-1] + pad[:, 1:])
            else: # Staggered in Y
                pad = F.pad(f.unsqueeze(0).unsqueeze(0), (0, 0, 1, 1), mode='constant', value=0).squeeze()
                return 0.5 * (pad[:-1, :] + pad[1:, :])

        f0 = torch.stack([center(self.u[:,:,idx], 'u'), center(self.v[:,:,idx], 'v'), self.t[:,:,idx], self.p[:,:,idx]], dim=0)
        f1 = torch.stack([center(self.u[:,:,idx+1], 'u'), center(self.v[:,:,idx+1], 'v'), self.t[:,:,idx+1], self.p[:,:,idx+1]], dim=0)
        
        return f0, f1, {'u': self.u[:,:,idx+1], 'v': self.v[:,:,idx+1], 'p': self.p[:,:,idx+1], 't': self.t[:,:,idx+1]}

    def get_params(self):
        return {**self.params, 'norm_params': self.norm_params, 'nanofluid_props': self.nanofluid_props}

    def get_nanofluid_properties(self):
        return self.nanofluid_props
