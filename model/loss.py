import torch
import torch.nn.functional as F

def ycbcr_weighted_mse(pred, target, weights=(1.0, 0.5, 0.5)):
    """
    pred, target: [B,C,H,W], C = YCbCr
    weights: tuple для (Y, Cb, Cr)
    """
    # [C] -> [1,C,1,1] для broadcasting
    w = torch.tensor(weights, device=pred.device).view(1, -1, 1, 1)
    loss = ((pred - target)**2) * w
    return loss.mean()

def ycbcr_mae_split(pred, target):
    """
    pred, target: [B,C,H,W], C = YCbCr
    """

    diff = (pred - target).abs()

    loss_y = diff[:, 0].mean()
    loss_cb = diff[:, 1].mean()
    loss_cr = diff[:, 2].mean()

    loss_cbcr = (loss_cb + loss_cr) / 2

    return loss_y, loss_cbcr



def scharr_filters(device, channels):
    kx = torch.tensor([
        [-3, 0, 3],
        [-10, 0, 10],
        [-3, 0, 3]
    ], dtype=torch.float32, device=device) / 16.0

    ky = kx.t()

    kx = kx.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    ky = ky.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)

    return kx, ky


def image_gradients(x):
    B, C, H, W = x.shape
    kx, ky = scharr_filters(x.device, C)

    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)

    return gx, gy


def gradient_loss(pred, target):
    gx_p, gy_p = image_gradients(pred)
    gx_g, gy_g = image_gradients(target)

    grad_loss = (
        F.l1_loss(gx_p, gx_g) +
        F.l1_loss(gy_p, gy_g)
    )
    return grad_loss


def laplacian_filter(device, channels):
    k = torch.tensor([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=torch.float32, device=device)

    k = k.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    return k


def laplacian(x):
    B, C, H, W = x.shape
    k = laplacian_filter(x.device, C)
    return F.conv2d(x, k, padding=1, groups=C)


def laplacian_loss(pred, target):
    lap_p = laplacian(pred)
    lap_g = laplacian(target)

    lap_loss = F.l1_loss(lap_p, lap_g)
    return lap_loss


def combined_mae_grad_lap_loss(pred, target, lambda_y, lambda_cbcr, lambda_grad, lambda_laplasian):
    mae = ycbcr_weighted_mae(pred, target, weights=(lambda_y, lambda_cbcr, lambda_cbcr))
    grad = gradient_loss(pred, target)
    lap = laplacian_loss(pred, target)

    total_loss = mae + lambda_grad * grad + lambda_laplasian * lap
    return total_loss


def rotations(x):
    return [
        x,
        x.rot90(1, (-2, -1)),
        x.rot90(2, (-2, -1)),
        x.rot90(3, (-2, -1)),
    ]


def invariant_tiny_loss(pred, target, p=3):
    losses = []

    for pr, gt in zip(rotations(pred), rotations(target)):
        g = gradient_loss(pr, gt)
        l = laplacian_loss(pr, gt)
        losses.append(g + l)

    losses = torch.stack(losses)  # [4]
    return (losses.pow(p).mean()).pow(1 / p)

def random_crop_pair(pred, target, size=3):
    B, C, H, W = pred.shape
    i = torch.randint(0, H - size + 1, (1,), device=pred.device)
    j = torch.randint(0, W - size + 1, (1,), device=pred.device)

    pred_crop = pred[..., i:i+size, j:j+size]
    tgt_crop  = target[..., i:i+size, j:j+size]
    return pred_crop, tgt_crop