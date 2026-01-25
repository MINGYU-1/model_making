import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDecoderCondVAE(nn.Module):
    def __init__(self,x_dim,c_dim,z_dim,z1_dim, h1=64, h2=32): #z1_dim은 다른 encoder에서 넣은값
        
        #z1은 surrogate만들떄 사용
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        
        ## encoder[x,c] 내에서 데이터를 넣는 방법
        self.encoder = nn.Sequential(
            nn.Linear(x_dim+c_dim,h2),
            nn.ReLU(),
            nn.Linear(h2,h1),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(h1,z_dim)
        self.logvar_head = nn.Linear(h1,z_dim)

        ## decoder_bce[z+c]->recon(x_dim)
        self.decoder_bce = nn.Sequential(
            nn.Linear(z_dim+c_dim,h1),
            nn.ReLU(),
            nn.Linear(h1,h2),
            nn.ReLU(),
            nn.Linear(h2,x_dim)
        )
        ## decoder_mse[z+c]->recon(x_dim)
        self.decoder_mse = nn.Sequential(
            nn.Linear(z_dim+c_dim,64),
            nn.ReLU(),
            nn.Linear(64,x_dim)
        )
        # 입력: z_dim + c_dim (촉매 정보와 반응 조건을 모두 고려)
        self.surrogate_head = nn.Sequential(
            nn.Linear(z_dim + c_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, c_dim)         )
    
    def reparameterize(self,mu,log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu +std*eps
    
    def forward(self,x,c):

        h = self.encoder(torch.cat([x, c], dim=1))
        z_mu = self.mu_head(h)
        z_logvar = self.logvar_head(h)
        z = self.reparameterize(z_mu, z_logvar)

        bce_logit = self.decoder_bce(torch.concat([z, c], dim=1))
        prob_mask = torch.sigmoid(bce_logit)
        
        # 학습 시에는 binary_out 대신 raw output을 사용하여 gradient 유지
        x_hat_raw = self.decoder_mse(torch.concat([z, c], dim=1))
        
        binary_out = (prob_mask > 0.5).float()
        x_hat = x_hat_raw * binary_out

        surrogate_input = torch.cat([z, c], dim=1)
        c_hat = self.surrogate_head(surrogate_input)

        return bce_logit, binary_out, x_hat, z_mu, z_logvar, c_hat
    
    def decode(self, z, c, threshold=0.5):
        dec_in = torch.cat([z, c], dim=1)

        bce_logit = self.decoder_bce(dec_in)
        prob_mask = torch.sigmoid(bce_logit)
        binary_out = (prob_mask > threshold).float()
        
        x_hat = self.decoder_mse(dec_in)
        x_hat *= prob_mask

        c_hat = self.surrogate_head(torch.cat([z, c], dim=1))
        return bce_logit, prob_mask, binary_out, x_hat, c_hat
    

   
    @torch.no_grad()
    def generate(self, c, n_samples=1, threshold=0.5):
   
        B = c.size(0)
        z = torch.randn(B * n_samples, self.z_dim, device=c.device)
        c_rep = c.repeat_interleave(n_samples, dim=0)

        bce_logit, prob_mask, binary_out, x_hat, c_hat = self.decode(
            z, c_rep, threshold=threshold)
        return x_hat, prob_mask, binary_out, c_hat