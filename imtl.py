import torch
import torch.nn as nn

from typing import Optional


__all__ = ['IMTL']


class IMTL(nn.Module):
    '''Backward must be called on all outputs'''
    def __init__(self, method: str='hybrid', num_losses: int=1, init_val: Optional[list]=None):
        super().__init__()
        self.method = method
        self.num_losses = -1
        init_loss = [1 for _ in range(num_losses)] if not isinstance(init_val, list) else [x for x in init_val]
        if 'gradient' in method:
            self.register_buffer('s_t', torch.tensor(init_loss))
        else:
            self.register_parameter('s_t', torch.tensor(init_loss))
        
    def forward(self, 
                shared_parameters,
                losses: list[torch.Tensor, ...],
               ) -> tuple[torch.Tensor, torch.Tensor]:
        # >>> Loss Balance
        L_t, g, u = [], [], []
        
        for i, l in enumerate(losses):
            if self.method in ['hybrid','loss']:
                # If 'hybrid' or 'loss' method selected, then learnable scale s_t is used
                L_t.append(l * self.e.pow(self.s_t[i]) - self.s_t[i])
            else:
                # Else, the losses are unscaled
                L_t.append(l)

            g[i].append(torch.flatten(torch.autograd.grad(L_t[-1], shared_parameters, retain_graph=True)))
            u[i].append(g[i] / torch.linalg.norm(g[i]))
            
        g_t = torch.stack([_g for _g in g])
        u_t = torch.stack([_u for _u in u])
       
        # >>> Gradient Balance
        D = g_t[0,...] - g_t[1:,...]
        U = u_t[0,...] - u_t[1:,...]

        arg0 = D.matmul(U.t())
        arg1 = torch.eye(arg0.shape[-1].data)
        alpha_2T = g_t[0,...].matmul(U.t()).matmul(torch.linalg.solve(arg0,arg1))
        
        alpha = torch.cat([torch.ones(1, device=self.device) - alpha_2T.sum(dim=0, keepdim=True), 
                           alpha_2T], dim=0).squeeze()
        
        # All methods return two outputs: gradient loss and individual losses
        if self.method=='hybrid':
            return torch.sum(L_t * alpha), L_t
        elif self.method=='gradient':
            return torch.sum(L_t * alpha), None
        elif self.method=='loss':
            return None, L_t
        
    def parameters(self) -> List[torch.Tensor]:
        '''return learnable parameters to be added to the optimizer'''
        if not 'gradient' in self.method:
            return [self.s_t]
        return []
    
