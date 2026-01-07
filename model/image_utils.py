from PIL import Image
import torch
import numpy as np

def rgb_to_ycbcr_tensor(img: torch.Tensor) -> torch.Tensor:
    '''
    :param img: RGB tensor in range [0; 1], shape [C,H,W] or [B,C,H,W]
    '''

    if img.ndim == 3:
        img = img.unsqueeze(0)  # add batch

    # BT.709
    matrix = torch.tensor([[0.2126, 0.7152, 0.0722],
                           [-0.1146, -0.3854, 0.5],
                           [0.5, -0.4542, -0.0458]], device=img.device)
    shift = torch.tensor([0., 0.5, 0.5], device=img.device)  # Cb Cr offset

    # [B,C,H,W] -> [B,H,W,C]
    img_perm = img.permute(0,2,3,1)
    ycbcr = torch.tensordot(img_perm, matrix.T, dims=1) + shift

    ycbcr = ycbcr * 2 - 1

    ycbcr = ycbcr.permute(0,3,1,2)
    if ycbcr.shape[0] == 1:
        ycbcr = ycbcr.squeeze(0) # remove batch if single image was passed
    return ycbcr

def ycbcr_to_rgb_tensor(ycbcr: torch.Tensor) -> torch.Tensor:
    '''
    :param ycbcr: YCbCr tensor in range [-1; 1]
    :return: RGB tensor in range [0; 1]
    '''

    # ycbcr: [-1,1]
    ycbcr = (ycbcr + 1) / 2  # [-1,1] -> [0,1]
    Y = ycbcr[:,0:1]
    Cb = ycbcr[:,1:2] - 0.5
    Cr = ycbcr[:,2:3] - 0.5

    # BT.709
    R = Y + 1.5748 * Cr
    G = Y - 0.1873 * Cb - 0.4681 * Cr
    B = Y + 1.8556 * Cb

    rgb = torch.cat([R,G,B], dim=1)
    rgb = rgb.clamp(0,1)
    return rgb



def ycbcr_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    t = tensor.clone()
    t = torch.clamp(t, -1.0, 1.0)

    arr = t.permute(1, 2, 0) # CxHxW -> HxWxC
    arr = (arr + 1.0) / 2.0 # [0; 1]
    arr = arr * 255 # [0; 255]
    arr = arr.cpu().numpy()
    arr = arr.astype(np.uint8)

    img = Image.fromarray(arr, mode="YCbCr")

    return img.convert("RGB")