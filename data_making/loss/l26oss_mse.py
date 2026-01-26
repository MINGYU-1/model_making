import torch
import torch.nn.functional as F
def integrated_loss_fn(x_hat, x, mu, logvar, 
                       alpha=1.0,gamma=0.01):


    mse_loss = F.mse_loss(x_hat, x, reduction='mean')

    # 3. KL Divergence: Latent Space 정규화
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl_per_sample.mean()

    total_loss = (alpha* mse_loss + gamma * kl_loss ) 
    return {
        'loss': total_loss,
        'mse_loss': mse_loss ,
        'kl_loss': kl_loss 
    }