import torch
import torch.nn as nn
import copy


__all__ = ['IMTL']


'''
input: list of raw losses
for each raw loss:
    L_t = 1*e**(s_t)*L_t^{raw} - s_t
    multiply the scaled task-sepcific losses by the gradient of the shared features - MSE/CEB???
    - g_t
    compute the unit-norm of the previous step
Next:
compute gradient differences between task 1 and all the others using g_t from above - D
compute the unit-norm gradient differences using unit-norm values from above - U
compute scaling factors for tasks 2 to T - alpha_{2:T} = g1*U.T * inv(D*U.T)
compute scaling factors for all tasks alpha = [1-ones.matmul(alpha_{2:T}.T), alpha_{2:T}]
update task-shared parameters - backward pass
for tasks 1 to T:
    update task-specific parameters - backward pass
    update loss scale parameters s_t
    
s_t are learnable parameters - need to be added to the optimizer!
    
'''

class IMTL(nn.Module):
    def __init__(self, method='hybrid'):
        super().__init__()
        self.method = method
        self.s_t = False
        self.num_losses = -1
        self.ind = []
        self.register_buffer('e', torch.exp(torch.ones([1])))
        
    def instantiate(self, 
                    device,
                    losses: list[torch.Tensor, ...]):
        del self.s_t
        self.device = data.device
        for i, loss in enumerate(losses):
            self.ind.append(-1)
            if loss.requires_grad: 
                self.num_losses += 1
                self.ind[-1] = self.num_losses
                
        self.register_parameter('s_t', nn.Parameter(torch.ones(self.num_losses+1).squeeze(), requires_grad=True))
        
    def forward(self, 
                data: torch.Tensor,
                losses: list[torch.Tensor, ...],
               ) -> tuple([torch.Tensor,...]):
        if isinstance(self.s_t, bool): self.instantiate(data.device, losses)
            
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
