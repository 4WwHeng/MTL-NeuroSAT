import torch
import torch.nn as nn
from core.Neurosat import NeuroSATNetwork

# MTL Model Definition with separate heads for ADM and CTP, plus an uncertainty-weighted loss function for dynamic task balancing

class MlpHeadModel(nn.Module):
    def __init__(self, in_dim=65, hid_dim=128, out_dim=2):
        super(MlpHeadModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim), 
            nn.ReLU(),
            
            nn.Linear(hid_dim, hid_dim // 2),
            nn.BatchNorm1d(hid_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hid_dim // 2, out_dim)
        )
        
    def forward(self, x):
        return self.network(x)
    

class MtlNeuroSAT(nn.Module):
    def __init__(self, d_model=64, T=26):
        super(MtlNeuroSAT, self).__init__()
        self.core = NeuroSATNetwork(d_model=d_model, T=T)
        self.adm = MlpHeadModel(in_dim=d_model+1, hid_dim=207, out_dim=2)
        self.ctp = MlpHeadModel(in_dim=d_model+1, hid_dim=220, out_dim=3)
        
    def forward(self, batch_data):
        """Unified forward pass for all three networks."""
        outputs, L_h, C_h, var_votes = self.core(batch_data)
        adm_logits = self.adm(L_h)
        ctp_logits = self.ctp(C_h)
        
        return outputs, adm_logits, ctp_logits


# uncertainty weights
class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, n_tasks=3):
        super().__init__()
        # log(sigma^2) for each task - learnable
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
    
    def forward(self, losses):
        total = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + 0.5 * self.log_vars[i]
        return total
