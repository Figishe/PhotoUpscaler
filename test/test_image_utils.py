from model.image_utils import *
import torch

def test_rgb_2_ycbcr_error() -> None:
    rgb = torch.rand(1,3,128,128)
    y = rgb_to_ycbcr_tensor(rgb)
    rgb2 = ycbcr_to_rgb_tensor(y)

    error = (rgb - rgb2).abs().max()
    assert error < 1e-5, f"RGB->YCbCr->RGB conversion error too high: {error}"