from PIL import Image
import torch
import numpy as np

_RGB_TO_YCBCR_709 = torch.tensor([
    [ 0.2126,  0.7152,  0.0722],
    [-0.114572, -0.385428,  0.5],
    [ 0.5, -0.454153, -0.045847],
])
_YCBCR_TO_RGB_709 = torch.linalg.inv(_RGB_TO_YCBCR_709)


def rgb_to_ycbcr_tensor(rgb_in: torch.Tensor) -> torch.Tensor:
    '''
    :param img: RGB tensor in range [0; 1], shape [C,H,W] or [B,C,H,W]
    '''
    if rgb_in.ndim == 3:
        img = rgb_in.unsqueeze(0)  # add batch
    else:
        img = rgb_in

    img = img.permute(0,2,3,1)
    
    M = _RGB_TO_YCBCR_709.to(device=img.device, dtype=img.dtype)
    ycbcr = torch.tensordot(img, M.T, dims=1)
    ycbcr[..., 1:] += 0.5  # Cb/Cr center
    ycbcr = ycbcr * 2 - 1  # [-1,1]
    ycbcr = ycbcr.permute(0,3,1,2)
    if rgb_in.ndim == 3:
        ycbcr = ycbcr.squeeze(0)  # remove batch
    return ycbcr


def ycbcr_to_rgb_tensor(ycbcr: torch.Tensor) -> torch.Tensor:
    '''
    :param ycbcr: YCbCr tensor in range [-1; 1]
    :return: RGB tensor in range [0; 1]
    '''
    ycbcr = (ycbcr + 1) / 2  # [0,1]
    ycbcr[:,1:] -= 0.5       # remove offset
    ycbcr = ycbcr.permute(0,2,3,1)

    M = _YCBCR_TO_RGB_709.to(device=ycbcr.device, dtype=ycbcr.dtype)
    rgb = torch.tensordot(ycbcr, M.T, dims=1)
    rgb = rgb.clamp(0,1)

    return rgb.permute(0,3,1,2)


def ycbcr_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    rgb = ycbcr_to_rgb_tensor(tensor.unsqueeze(0))[0]
    rgb = (rgb * 255).round().clamp(0,255).byte()
    return Image.fromarray(rgb.permute(1,2,0).cpu().numpy(), "RGB")


if __name__ == "__main__":
    rgb = torch.rand(1,3,128,128, device="cuda")
    y = rgb_to_ycbcr_tensor(rgb)
    rgb2 = ycbcr_to_rgb_tensor(y)

    print(f"rgb-ycbcr conversion error = {(rgb - rgb2).abs().max()}")
