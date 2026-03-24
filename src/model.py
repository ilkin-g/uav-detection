import torch
import torch.nn as nn
import torch.nn.functional as F
from src.wavelets import adaRatGaussWav

class UAVDetector(nn.Module):
    def __init__(self, signal_length, m_coeffs=20, p_zeros=3, r_poles=4, kernel_length=128, device=None):
        super(UAVDetector, self).__init__()
        self.device = device if device else torch.device("cpu")
        self.m_coeffs = m_coeffs
        self.p_zeros = p_zeros
        self.r_poles = r_poles
        
        self.kernel_length = kernel_length
        
        self.rgw_zeros = nn.Parameter(torch.randn(p_zeros))
        self.rgw_poles = nn.Parameter(torch.randn(2 * r_poles))
        self.rgw_scales = nn.Parameter(torch.rand(m_coeffs))
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(m_coeffs),
            nn.Dropout(p=0.5),
            nn.Linear(m_coeffs, 15),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(15, 3) 
        )

    def rgw_conv(self, x):
        t_translations = torch.zeros(self.m_coeffs, device=self.device) 
        aff = torch.stack((self.rgw_scales, t_translations), dim=1).reshape(-1)
        par = torch.cat([self.rgw_zeros, self.rgw_poles, aff, torch.tensor([2.0], device=self.device)])

        Phi = adaRatGaussWav(self.kernel_length, self.m_coeffs, par, self.p_zeros, self.r_poles, a=-5.0, b=5.0, bmin=0.01, s_square=True, device=self.device)[0]
        
        kernels = Phi.T.unsqueeze(1) 
        
        conv_out = F.conv1d(x, kernels)
        return conv_out

    def forward(self, x):

        features = self.rgw_conv(x)
        
        pooled_features = F.adaptive_max_pool1d(features, 1).squeeze(-1)
        
        output = self.classifier(pooled_features)
        
        return output