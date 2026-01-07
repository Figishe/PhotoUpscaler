import torch

def ycbcr_weighted_mse(pred, target, weights=(1.0, 0.5, 0.5)):
    """
    pred, target: [B,C,H,W]
    weights: tuple для (Y, Cb, Cr)
    """
    # [C] -> [1,C,1,1] для broadcasting
    w = torch.tensor(weights, device=pred.device).view(1, -1, 1, 1)
    loss = ((pred - target)**2) * w
    return loss.mean()
