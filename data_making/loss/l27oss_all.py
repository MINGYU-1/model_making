import torch
import torch.nn.functional as F
#x,x2,x3은 그냥 입력값
def integrated_loss_fn(binary_logit, x_hat,x2_hat,x3_hat, x,x2,x3, mu, logvar,mu2,logvar2,mu3,logvar3, a=1.0,b=1.0, c=1.0,d =1.0,e=1.0,f=1.0,g=1.0):
    
    batch_size = x.shape[0]

    # 1. Classification Loss (BCE): 금속 존재 여부 (이미지의 probability 부분)
    x_binary = (x > 0).float()
    bce_loss = F.binary_cross_entropy_with_logits(binary_logit, x_binary, reduction='sum')

    # 2. Reconstruction Loss (MSE): 금속의 수치 정보
    # 실제 존재하는 부분에 대해서만 MSE를 계산하는 것이 더 정확할 수 있습니다.
    mse_loss = F.mse_loss(x_hat, x, reduction='sum')
    mse2_loss = F.mse_loss(x2_hat, x2, reduction='sum')
    mse3_loss = F.mse_loss(x3_hat, x3, reduction='sum')

    # 3. KL Divergence: Latent Space 정규화
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl2_loss = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
    kl3_loss = -0.5 * torch.sum(1 + logvar3 - mu3.pow(2) - logvar3.exp())


    # 최종 손실 합산 (가중치 조절)
    # 각 loss를 batch_size로 나누어 평균 손실을 구함
    total_loss = (a* mse_loss + b*mse2_loss+ c*mse3_loss + d * bce_loss + e * kl_loss+f*kl2_loss + g*kl3_loss) / batch_size

    return {
        'loss': total_loss,
        'bce_loss': bce_loss / batch_size,
        'mse_loss': mse_loss / batch_size,
        'mse2_loss':mse2_loss/batch_size,
        'mse3_loss':mse3_loss/batch_size,
        'kl_loss': kl_loss / batch_size,
        'kl2_loss':kl2_loss/batch_size,
        'kl3_loss':kl3_loss/batch_size
    }