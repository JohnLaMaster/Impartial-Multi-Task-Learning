import torch
import torch.nn as nn


__all__ = ['IMTL']


class IMTL(nn.Module):
    def __init__(self, method: str='hybrid', num_losses: int=1):
        super().__init__()
        self.method = method
        self.first = True
        self.num_losses = -1
        self.ind = []
        self.register_buffer('e', torch.exp(torch.ones([1])))
        self.register_parameter('s_t', torch.ones(num_losses+1).squeeze())
        
    def instantiate(self, 
                    device,
                    losses: list[torch.Tensor, ...]):
        self.device = data.device
        for i, loss in enumerate(losses):
            self.ind.append(-1)
            if loss.requires_grad: 
                self.num_losses += 1
                self.ind[-1] = self.num_losses
        self.first = False
        
    def forward(self, 
                data: torch.Tensor,
                losses: list[torch.Tensor, ...],
               ) -> tuple([torch.Tensor,...]):
        if isinstance(self.first, True): self.instantiate(data.device, losses)
            
        # >>> Loss Balance
        L_t = torch.empty(self.s_t.numel(), device=self.device)
        g_t = torch.empty([self.num_losses+1, 1, data.shape[-1]], device=self.device)
        for i, loss in zip(self.ind, losses):
            if loss.requires_grad:
                L_t[i] = loss * self.e.pow(self.s_t[i]) - self.s_t[i]
                g_t[i,::] = torch.autograd.grad(L_t[i], data, retain_graph=True, create_graph=True)[0].mean(dim=0, keepdim=True)
        u_t = g_t / (torch.linalg.norm(g_t, 2, (-1,-2)) + 1e-6).unsqueeze(-1).unsqueeze(-1)
       
        # >>> Gradient Balance
        D = g_t[0,::].unsqueeze(0).repeat(self.num_losses,1,1) - g_t[1:,::]
        UT = u_t[0,:].unsqueeze(0).repeat(self.num_losses,1,1).mT - u_t[1:,::].mT

        alpha_2T = g_t[0,::].unsqueeze(0).matmul(UT).matmul(torch.linalg.pinv(D.matmul(UT)))
        alpha = torch.cat([torch.ones([1, alpha_2T.shape[1], alpha_2T.shape[2]], device=self.device) - 
                           alpha_2T.sum(dim=0, keepdim=True), alpha_2T], dim=0).squeeze()
        
        if self.method=='hybrid':
            return torch.sum(L_t * alpha), [loss.backward(retain_graph=True) for i, loss in enumerate(L_t)]
        elif self.method=='gradient':
            return torch.sum(L_t * alpha)
        elif self.method=='loss':
            return [loss.backward(retain_graph=True) for i, loss in enumerate(L_t)]
