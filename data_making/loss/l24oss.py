import torch
import torch.nn.functional as F
def integrated_loss_fn(binary_logit,x_hat, x, mu, logvar,beta = 1.0):
    x_binary = (x>0).float()
    bce_loss = F.binary_cross_entropy_with_logits(binary_logit, x_binary,reduction= 'sum')
    mse_elements = (x_hat-x)**2
    mse_loss = torch.sum(mse_elements)
    kl_loss = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    batch_size = x.shape[0]
    loss = (mse_loss+ beta*kl_loss)/batch_size


    return {
        'loss': loss,
        'bce': mse_loss/batch_size,
        'kl_loss':kl_loss/batch_size
    }

