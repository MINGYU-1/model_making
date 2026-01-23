import torch
import torch.nn as nn
import torch.nn.functional as F

## predictor 구하는방법
class MultiDecoderCondVAE(nn.Module):
    def __init__(self,x_dim,c_dim,z_dim,z1_dim, h1=32, h2=64): #z1_dim은 다른 encoder에서 넣은값
        
        #z1은 surrogate만들떄 사용
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        
        ## encoder[x,c] 내에서 데이터를 넣는 방법
        self.encoder = nn.Sequential(
            nn.Linear(x_dim+c_dim,h1),
            nn.ReLU(),
            nn.Linear(h1,h2),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(h2,z_dim)
        self.logvar_head = nn.Linear(h2,z_dim)

        ## decoder_bce[z+x]->recon(x_dim)
        self.decoder_bce = nn.Sequential(
            nn.Linear(z_dim+x_dim,h2),
            nn.ReLU(),
            nn.Linear(h2,h1),
            nn.ReLU(),
            nn.Linear(h1,x_dim)
        )
        ## decoder_mse[z+c]->recon(x_dim)
        self.decoder_mse = nn.Sequential(
            nn.Linear(z_dim+c_dim,128),
            nn.ReLU(),
            nn.Linear(128,x_dim)
        )
    def reparameterize(self,mu,log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu +std*eps
    
    def forward(self,x,c,hard_mask=False):

        ### 관련해서 z_mu,z_logvar을 활용해서 값을 구하기
        h = self.encoder(torch.cat([x,c],dim = 1))
        z_mu = self.mu_head(h)
        z_logvar = self.logvar_head(h)
        z = self.reparameterize(z_mu,z_logvar)

        ### mask_logits을 구하는데 있어서 값 넣기
        mask_logits = self.decoder_bce(torch.concat([z,x],dim = 1))
        prob_mask = torch.sigmoid(mask_logits) ## 확률적 해석
        ## mask_out을 구할때의 방식에 대해 이렇게 구한 값이 0보다 크면 
        mask_out = (prob_mask>0.5).float() ## 결정적 이진 출력
        recon_numeric = self.decoder_mse(torch.concat([z,c],dim = 1))

        return mask_logits, prob_mask, mask_out, recon_numeric,z_mu,z_logvar
    
def integrated_loss_fn(mask_logits,recon_numeric, target_x, mu, logvar, beta = 1.0):
    target_mask = (target_x>0).float()
    bce_loss = F.binary_cross_entropy_with_logits(mask_logits, target_mask,reduction= 'sum')
    mse_elements= (recon_numeric-target_x)**2
    masked_mse_loss= torch.sum(mse_elements*target_mask)
    kl_loss = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    batch_size = target_x.shape[0]
    loss = (masked_mse_loss+beta*kl_loss)/batch_size


    return {
        'loss': loss,
        'mse': masked_mse_loss/batch_size,
        'kl_loss':kl_loss/batch_size
    }


