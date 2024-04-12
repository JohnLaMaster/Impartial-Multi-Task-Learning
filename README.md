# Impartial Multi-Task Learning
PyTorch implementation of "Towards Impartial Multi-Task Learning"

"Towards Impartial Multi-task Learning". 
Liyang Liu, Yi Li, Zhanghui Kuang, Jing-Hao Xue, Yimin Chen, Wenming Yang, Qingmin Liao, Wayne Zhang

OpenReview: https://openreview.net/forum?id=IMPnRXEWpvr

Source code written by: Ing. John T LaMaster


# Implementation
1. Instantiate the module and send to the GPU. As described in the paper, the possible methods are 'gradient', 'loss', and 'hybrid'.
2. Use the function "itertools.chain()" to combine the parameters of the NN and the IMTL module when defining the optimizer
                
        num_losses = 5 # for example
        init_values = None # or a list of initial scaling values
        self.IMTL = IMTL(method='hybrid', num_losses=num_losses, init_loss=init_values).to(device)
        if self.opt.IMTL: 
            parameters = itertools.chain(self.network.parameters(), self.IMTL.parameters())
        else:
            self.network.parameters()
        self.optimizer = torch.optim.Adam(parameters, lr=opt.lr, betas=(opt.beta1, opt.beta2))

4. You will need to modify the NN code to return the intermediate feature *z*. This is the output of the encoder before the global pooling, flattening, and linear layers.
5. During training, append each *task*, <img src="https://render.githubusercontent.com/render/math?math=L_{t}^{raw}">, to a list *loss*.
**Note:** There are circumstances in which not all losses should be included in IMTL. For such cases, these values can be popped from the *loss* list before calling IMTL. As shown below, do not forget to call their backward call.
6. To evaluate the effect of each objective function, I often turn some off. To handle this, the IMTL code will only use tensors with *requires_grad=True*. These values do _NOT_ need to be removed. The list and the intermediate feature *z* can now be used to call *self.IMTL()*.
                                
        # For tracking the loss and not using IMTL
        ind = []
        for i, cond, l in enumerate(zip(exclude,loss)): 
            if cond:
                other_loss += l
                ind.append(i)
        for i in reverse(ind): 
            loss.pop(i)
   
        # For using IMTL
        if self.IMTL and self.opt.phase=='train': 
            # shared, specific = grad_loss, scaled_losses
            grad_loss, scaled_losses = self.IMTL.backward(shared_parameters=[model.parameters()], losses=loss)
            other_loss.backward() # If calculating losses that bypass IMTL
            
8. As long as IMTL.backwards() is used instead of the forward pass, nothing needs to be done to *shared* or *specific* as their .backward() calls are used in self.IMTL()
9. Finally, the optimizer can make it's forward step. 
        
        self.optimizer.step()
        self.optimizer.zero_grad()

