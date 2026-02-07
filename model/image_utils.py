from PIL import Image
import torch
import numpy as np

# https://web.archive.org/web/20120403123714/http://www.equasys.de/colorconversion.html
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


def ycbcr_to_rgb_tensor(ycbcr_in: torch.Tensor) -> torch.Tensor:
    '''
    :param ycbcr: YCbCr tensor in range [-1; 1]
    :return: RGB tensor in range [0; 1]
    '''
    if ycbcr_in.ndim == 3:
        ycbcr = ycbcr_in.unsqueeze(0)  # add batch
    else:
        ycbcr = ycbcr_in

    ycbcr = (ycbcr + 1) / 2  # [0,1]
    ycbcr[:,1:] -= 0.5       # remove offset
    ycbcr = ycbcr.permute(0,2,3,1)

    M = _YCBCR_TO_RGB_709.to(device=ycbcr.device, dtype=ycbcr.dtype)
    rgb = torch.tensordot(ycbcr, M.T, dims=1)
    rgb = rgb.clamp(0,1)
    rgb = rgb.permute(0,3,1,2)

    if ycbcr_in.ndim == 3:
        rgb = rgb.squeeze(0)  # remove batch

    return rgb


def ycbcr_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    rgb = ycbcr_to_rgb_tensor(tensor)
    rgb = (rgb * 255).round().clamp(0,255).byte()
    return Image.fromarray(rgb.permute(1,2,0).cpu().numpy(), "RGB")


def pil_to_lpips_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    rgb = torch.from_numpy(np.array(img)).to(device)
    rgb = rgb.permute(2,0,1).unsqueeze(0).float() / 255.0
    return rgb
