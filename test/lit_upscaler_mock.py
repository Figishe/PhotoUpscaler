import torch
import torch.nn.functional as F
import lightning as L

class LitUpscalerMock(L.LightningModule):

    def __init__(self, upscale_factor: int = 2) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            scale_factor=self.upscale_factor,
            mode="bicubic",
            align_corners=False
        )