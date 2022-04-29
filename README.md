# Impartial_Multi-Task_Learning
PyTorch implementation of "Towards Impartial Multi-Task Learning"

"Towards Impartial Multi-task Learning". 
Liyang Liu, Yi Li, Zhanghui Kuang, Jing-Hao Xue, Yimin Chen, Wenming Yang, Qingmin Liao, Wayne Zhang

OpenReview: https://openreview.net/forum?id=IMPnRXEWpvr

Source code written by: Ing. John T LaMaster


# Implementation
1. Instantiate the module and send to the GPU. As described in the paper, the possible methods are 'gradient', 'loss', and 'hybrid'.
2. Use the function "itertools.chain()" to combine the parameters of the NN and the IMTL module when defining the optimizer
                
        self.IMTL = IMTL(method='hybrid').to(device)
        if self.opt.IMTL: 
            parameters = itertools.chain(self.network.parameters(), self.IMTL.parameters())
        else:
            self.network.parameters()
        self.optimizer = torch.optim.Adam(parameters, lr=opt.lr, betas=(opt.beta1, opt.beta2))

3. You will need to modify the NN code to return the intermediate feature *z*. This is the output of the encoder before the global pooling, flattening, and linear layers.
4. During training, append each *task*, <img src="https://render.githubusercontent.com/render/math?math=L_{t}^{raw}">, to a list *loss*.
**Note:** There are circumstances in which not all losses should be included in IMTL. For such cases, the following code can be used. Define a list *index* to track which values in *loss* should be excluded. These losses need to be saved in order to call .backward() later. After summing, these values can be popped from the *loss* list.
5. To evaluate the effect of each objective function, I often turn some off. To handle this, the IMTL code will only use tensors with *requires_grad=True*. These values do _NOT_ need to be removed. The list and the intermediate feature *z* can now be used to call *self.IMTL()*.
                                
        for l in loss: 
            # For tracking the loss and not using IMTL
            self.loss += l
            
        if self.IMTL and self.opt.phase=='train': 
            length = len(loss)
            other_loss = 0
            for i, v in enumerate(reversed(index)):
                if v==1: 
                    other_loss += loss[length - 1 - i]
                    loss.pop(length - 1 - i)
            shared, specific = self.IMTL(self.intermediate, loss)
            other_loss.backward()
            
6. Nothing needs to be done to *shared* or *specific* as their .backward() calls are used in self.IMTL()
7. Finally, the optimizer can make it's forward step. The if-statement is for not using IMTL.
        
        if not (self.IMTL and self.opt.phase=='train'): 
            self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

