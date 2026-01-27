import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDecoderCondVAE(nn.Module):
    def __init__(self,x_dim,x2_dim,x3_dim, c_dim, z_dim,z2_dim,z3_dim, h1=32, h2=64): #z1_dim은 다른 encoder에서 넣은값
        
        #z1은 surrogate만들떄 사용
        super().__init__()
        self.x_dim = x_dim
        self.x2_dim = x2_dim
        self.x3_dim = x3_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.z2_dim = z2_dim
        self.z3_dim = z3_dim
        
        ## encoder[x,c] 내에서 데이터를 넣는 방법
        self.encoder = nn.Sequential(
            nn.Linear(x_dim+c_dim,h2),
            nn.ReLU(),
            nn.Linear(h2,h1),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(h1,z_dim)
        self.logvar_head = nn.Linear(h1,z_dim)

        ## encoder2[x2,z]->[z1]
        self.encoder2 = nn.Sequential(
            nn.Linear(x2_dim+z_dim,h2),
            nn.ReLU(),
            nn.Linear(h2,h1),
            nn.ReLU()
        )
        self.mu2_head = nn.Linear(h1,z2_dim)
        self.logvar2_head = nn.Linear(h1,z2_dim)

          ## encoder3[x3,z2]->[z3]
        self.encoder3 = nn.Sequential(
            nn.Linear(x3_dim+z2_dim,h2),
            nn.ReLU(),
            nn.Linear(h2,h1),
            nn.ReLU()
        )
        self.mu3_head = nn.Linear(h1,z3_dim)
        self.logvar3_head = nn.Linear(h1,z3_dim)     
       
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
            nn.Linear(z_dim+c_dim,h1),
            nn.ReLU(),
            nn.Linear(h1,h2),
            nn.ReLU(),
            nn.Linear(h2,x_dim)
        )

        ## decoder2_mse[x2+z2]->recon(x2_dim)
        self.decoder2_mse = nn.Sequential(
            nn.Linear(x2_dim+z2_dim,h1),
            nn.ReLU(),
            nn.Linear(h1,h2),
            nn.ReLU(),
            nn.Linear(h2,x2_dim)
        )
        ## decoder3_mse[x3+z3]->recon(x3_dim)
        self.decoder3_mse = nn.Sequential(
            nn.Linear(x3_dim+z3_dim,h1),
            nn.ReLU(),
            nn.Linear(h1,h2),
            nn.ReLU(),
            nn.Linear(h2,x3_dim)
        )
    #첫번째 z
    def reparameterize(self,mu,log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu +std*eps

    # 두번쨰 z
    def reparameterize2(self,mu2,log2_var):
        std2 = torch.exp(0.5*log2_var)
        eps2 = torch.randn_like(std2)
        return mu2 +std2*eps2

    #세번째 z
    def reparameterize3(self,mu3,log3_var):
        std3 = torch.exp(0.5*log3_var)
        eps3 = torch.randn_like(std3)
        return mu3 +std3*eps3
 
    def forward(self,x,x2,x3,c,z,z2):

        ### 관련해서 z_mu,z_logvar을 활용해서 값을 구하기
        h = self.encoder(torch.cat([x,c],dim = 1))
        z_mu = self.mu_head(h)
        z_logvar = self.logvar_head(h)
        z = self.reparameterize(z_mu,z_logvar)

        bce_logit = self.decoder_bce(torch.concat([z,c],dim = 1))
        prob_mask = torch.sigmoid(bce_logit) ## 확률적 해석
        ## mask_out을 구할때의 방식에 대해 이렇게 구한 값이 0보다 크면 
        binary_out = (prob_mask>0.5).float() ## 결정적 이진 출력
        x_hat = self.decoder_mse(torch.concat([z,c],dim = 1))
        x_hat 
        ## c_에 대한 예측

        # [x2,z]->mu
        h = self.encoder2(torch.cat([x2,z],dim = 1))
        z2_mu = self.mu2_head(h)
        z2_logvar = self.logvar2_head(h)
        z2 = self.reparameterize(z2_mu,z2_logvar)
        x2_hat = self.decoder2_mse(torch.concat([z2,x2],dim = 1))

        # [x3,z2]->mu
        h = self.encoder2(torch.cat([x3,z2],dim = 1))
        z3_mu = self.mu3_head(h)
        z3_logvar = self.logvar3_head(h)
        z3 = self.reparameterize(z3_mu,z3_logvar)
        x3_hat = self.decoder2_mse(torch.concat([z3,x2],dim = 1))

        return bce_logit ,binary_out, x_hat,x2_hat,x3_hat, z_mu,z_logvar
