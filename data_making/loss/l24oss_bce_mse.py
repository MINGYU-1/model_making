import torch
import torch.nn.functional as F
history = {'alpha': 0.5380115038688167, 'beta': 0.05270001399718565, 'gamma': 7.8997494750531745}
def integrated_loss_fn(binary_logit,x_hat, x, mu, logvar):
    x_binary = (x>0).float()
    bce_loss = F.binary_cross_entropy_with_logits(binary_logit, x_binary,reduction= 'sum')
    mse_elements= (x_hat-x)**2
    mse_loss= torch.sum(mse_elements*target_mask)
    kl_loss = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    batch_size = x.shape[0]
    loss = (history['alpha'] * mse_loss + history['beta']* bce_loss+ history['gamma']*kl_loss)/batch_size


    return {
        'loss': loss,
        'bce': bce_loss/batch_size,
        'mse': mse_loss/batch_size,
        'kl_loss':kl_loss/batch_size
    }


