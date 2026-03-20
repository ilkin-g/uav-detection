import torch
import torch.nn as nn
from src.rgw_layer import vp_layer
from src.wavelets import adaRatGaussWav

class UAVDetector(nn.Module):
    def __init__(self, signal_length, m_coeffs=10, p_zeros=3, r_poles=4, device=None):
        super(UAVDetector, self).__init__()
        self.device = device if device else torch.device("cpu")
        
        init_zeros = [0.5] * p_zeros
        init_poles = [0.1, 0.5] * r_poles
        init_sigma = [1.0]
        
        init_wavelets = []
        translations = torch.linspace(-4, 4, m_coeffs).tolist()
        for i in range(m_coeffs):
            init_wavelets.extend([1.0, translations[i]])
            
        initial_params = init_zeros + init_poles + init_wavelets + init_sigma
        init_tensor = torch.tensor(initial_params, dtype=torch.float32, device=self.device)
        nparams = len(initial_params)

        self.rgw_vp_layer = vp_layer(
            ada=adaRatGaussWav,
            n_in=signal_length,
            n_out=m_coeffs,
            nparams=nparams,
            p=p_zeros,
            r=r_poles,
            b_min=0.01,
            a=-5.0,
            b=5.0,
            penalty=0.1,
            target=0,
            device=self.device,
            init=init_tensor
        )
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(m_coeffs),
            nn.Dropout(p=0.5),
            nn.Linear(m_coeffs, 15),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(15, 3)
        )

    def forward(self, x):
        features = self.rgw_vp_layer(x)
        
        features = features.squeeze(-1) 
        
        output = self.classifier(features)
        
        return output