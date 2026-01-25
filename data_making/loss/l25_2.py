import torch
import torch.nn.functional as F
history = { 'alpha': 0.11702920706785994, 'beta': 0.07012716830407131, 'gamma': 9.19546202126989, 'surr': 7.816033477711758}
def integrated_loss_fn(bce_logit, x_hat, x, mu, logvar, c_hat, c):
    """
    Args:
        mask_logits: Decoder BCE의 출력 (Probability 예측용)
        x_hat: Decoder MSE의 출력 (Active Metal Feature 예측용)
        x: 실제 Active Metal 데이터
        mu, logvar: Latent space의 파라미터
        c_hat: Surrogate Predictor의 예측값 (CH4 Conversion)
        c: 실제 CH4 Conversion 값
        w_surr: Surrogate loss의 가중치 (보통 회귀 성능을 높이기 위해 크게 잡음)
    """
    batch_size = x.shape[0]

    # 1. Classification Loss (BCE): 금속 존재 여부 (이미지의 probability 부분)
    target_mask = (x > 0).float()
    bce_loss = F.binary_cross_entropy_with_logits(bce_logit, target_mask, reduction='sum')

    # 2. Reconstruction Loss (MSE): 금속의 수치 정보
    # 실제 존재하는 부분에 대해서만 MSE를 계산하는 것이 더 정확할 수 있습니다.
    mse_loss = F.mse_loss(x_hat, x, reduction='sum')

    # 3. KL Divergence: Latent Space 정규화
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 4. Surrogate Loss (MSE): CH4 Conversion 예측 성능
    # 이미지의 가장 우측 하단 예측값에 대한 Loss
    surr_loss = F.mse_loss(c_hat, c, reduction='sum')

    # 최종 손실 합산 (가중치 조절)
    # 각 loss를 batch_size로 나누어 평균 손실을 구함
    total_loss = (history['alpha']* bce_loss + history['beta'] * mse_loss + history['gamma'] * kl_loss + history['w_surr'] * surr_loss) / batch_size

    return {
        'loss': total_loss,
        'bce_loss': bce_loss / batch_size,
        'mse_loss': mse_loss / batch_size,
        'kl_loss': kl_loss / batch_size,
        'surr_loss': surr_loss / batch_size
    }