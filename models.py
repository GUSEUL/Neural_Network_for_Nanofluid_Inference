import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class STNNN(nn.Module):
    """
    Spatiotemporal Neural Network (STNNN) - Physics-informed Convolutional-Recurrent Network.
    Compatible with existing checkpoints.
    """
    def __init__(self, input_ch=4, output_ch=4, hidden=192, upscale=1, dropout_rate=0.2):
        super().__init__()
        
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(input_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        
        # ConvLSTM layers
        self.conv_lstm = DeepConvLSTM(hidden, hidden, num_layers=3, kernel_size=5, padding=2)
        
        # Residual block
        self.residual_block = ResidualBlock(hidden, hidden)
        
        # Decoder
        self.dec = nn.Sequential(
            nn.Conv2d(hidden, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(dropout_rate/2),
            nn.Conv2d(64, output_ch*(upscale**2), 3, padding=1),
            nn.PixelShuffle(upscale) if upscale > 1 else nn.Identity()
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        if x.dim() == 5:
            B, S, C, H, W = x.shape
            z = self.enc(x.view(B*S, C, H, W)).view(B, S, -1, H, W)
            z, _ = self.conv_lstm(z)
            z = self.residual_block(z[:, -1])
        else:
            z = self.enc(x)
            z = z.unsqueeze(1)
            z, _ = self.conv_lstm(z)
            z = self.residual_block(z.squeeze(1))
        return self.dec(z)

    def forward_with_latent(self, x):
        if x.dim() == 5:
            B, S, C, H, W = x.shape
            z = self.enc(x.view(B*S, C, H, W)).view(B, S, -1, H, W)
            z, _ = self.conv_lstm(z)
            latent = self.residual_block(z[:, -1])
        else:
            z = self.enc(x)
            z = z.unsqueeze(1)
            z, _ = self.conv_lstm(z)
            latent = self.residual_block(z.squeeze(1))
        return latent, latent.size(1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
    
    def forward(self, x):
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(res)
        return F.relu(out)

class DeepConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, kernel_size=5, padding=2):
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            ConvectionLSTM(in_channels if i==0 else hidden_channels, hidden_channels, kernel_size, padding)
            for i in range(num_layers)
        ])
        self.attention = nn.Conv2d(hidden_channels, 1, 1)
    
    def forward(self, x, states=None):
        if states is None: states = [None] * self.num_layers
        out = x
        new_states = []
        for i, cell in enumerate(self.cells):
            out, s = cell(out, states[i])
            new_states.append(s)
        last_out = out[:, -1]
        attn = torch.sigmoid(self.attention(last_out))
        out_final = out.clone(); out_final[:, -1] = last_out * attn
        return out_final, new_states

class ConvectionLSTM(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=5, padding=2):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, hidden_ch * 4, kernel_size, padding=padding)
        
        # Restoring LayerNorm for checkpoint compatibility
        self.layer_norm = nn.LayerNorm([hidden_ch, 42, 42])
        
        self.w_ci = nn.Parameter(torch.zeros(1, hidden_ch, 1, 1))
        self.w_cf = nn.Parameter(torch.zeros(1, hidden_ch, 1, 1))
        self.w_co = nn.Parameter(torch.zeros(1, hidden_ch, 1, 1))

    def forward(self, x, state=None):
        B, T, C, H, W = x.size()
        h, c = state if state else (torch.zeros(B, self.hidden_ch, H, W, device=x.device), torch.zeros(B, self.hidden_ch, H, W, device=x.device))
        out = torch.zeros(B, T, self.hidden_ch, H, W, device=x.device)
        for t in range(T):
            gates = self.conv(torch.cat([x[:, t], h], dim=1))
            i, f, g, o = torch.chunk(gates, 4, dim=1)
            i = torch.sigmoid(i + self.w_ci * c)
            f = torch.sigmoid(f + self.w_cf * c)
            c = f * c + i * torch.tanh(g)
            o = torch.sigmoid(o + self.w_co * c)
            h = o * torch.tanh(c)
            
            # Apply LayerNorm if dimensions match (as in original code)
            if H == 42 and W == 42:
                h = self.layer_norm(h)
                
            out[:, t] = h
        return out, (h, c)

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
    Surrogate model with FiLM conditioning for 4 parameters (Ra, Ha, Q, Da).
    Optimized for high-precision spatiotemporal sequence prediction.
    """
    PARAM_RANGES = {
        'Ra': (100.0, 1e8),
        'Ha': (0.0, 100.0),
        'Q': (-10.0, 10.0),
        'Da': (0.001, 0.15),
    }

    def __init__(self, input_ch=4, output_ch=4, hidden=192, use_film=True):
        super().__init__()
        self.use_film = use_film
        self.hidden_channels = hidden
        
        # 4 extra channels for parameter projection if not using FiLM (legacy support)
        # However, we prefer FiLM for sensitivity.
        self.stnnn = STNNN(input_ch=input_ch + (0 if use_film else 4), 
                          output_ch=output_ch, hidden=hidden)
        
        self.param_encoder = nn.Sequential(
            nn.Linear(4, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128)
        )
        
        if self.use_film:
            self.film_layer = FiLMLayer(conditioning_dim=128, feature_channels=hidden)
            self.film_decoder = nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(hidden, output_ch, 1)
            )

    def normalize_params(self, ra, ha, q, da):
        log_ra = torch.log10(torch.clamp(ra, min=1.0))
        ra_n = (log_ra - np.log10(self.PARAM_RANGES['Ra'][0])) / (np.log10(self.PARAM_RANGES['Ra'][1]) - np.log10(self.PARAM_RANGES['Ra'][0]))
        ha_n = (ha - self.PARAM_RANGES['Ha'][0]) / (self.PARAM_RANGES['Ha'][1] - self.PARAM_RANGES['Ha'][0])
        q_n = (q - self.PARAM_RANGES['Q'][0]) / (self.PARAM_RANGES['Q'][1] - self.PARAM_RANGES['Q'][0])
        log_da = torch.log10(torch.clamp(da, min=1e-5))
        da_n = (log_da - np.log10(self.PARAM_RANGES['Da'][0])) / (np.log10(self.PARAM_RANGES['Da'][1]) - np.log10(self.PARAM_RANGES['Da'][0]))
        return ra_n, ha_n, q_n, da_n

    def forward(self, x_seq, ra, ha, q, da):
        B, S, C, H, W = x_seq.shape
        ra_n, ha_n, q_n, da_n = self.normalize_params(ra, ha, q, da)
        param_vec = torch.stack([ra_n, ha_n, q_n, da_n], dim=-1) # [B, 4]
        
        if self.use_film:
            # Pass through STNNN until residual block
            # We need to expose the latent features from STNNN
            # For simplicity in this implementation, we'll re-implement the STNNN forward here
            # to inject FiLM at the right place (after ConvLSTM, before Decoder)
            
            x_reshaped = x_seq.view(B*S, C, H, W)
            z_reshaped = self.stnnn.enc(x_reshaped)
            z_seq = z_reshaped.view(B, S, -1, H, W)
            z_lstm, _ = self.stnnn.conv_lstm(z_seq)
            latent = self.stnnn.residual_block(z_lstm[:, -1])
            
            param_embed = self.param_encoder(param_vec)
            z_modulated = self.film_layer(latent, param_embed)
            output = self.film_decoder(z_modulated)
        else:
            # Legacy: concat params as channels
            ra_ch = ra_n.view(B, 1, 1, 1, 1).expand(B, S, 1, H, W)
            ha_ch = ha_n.view(B, 1, 1, 1, 1).expand(B, S, 1, H, W)
            q_ch = q_n.view(B, 1, 1, 1, 1).expand(B, S, 1, H, W)
            da_ch = da_n.view(B, 1, 1, 1, 1).expand(B, S, 1, H, W)
            x_input = torch.cat([x_seq, ra_ch, ha_ch, q_ch, da_ch], dim=2)
            output = self.stnnn(x_input)
            
        return output
