import lightning as L
import torch
from model.upscaler import SuperResNet
from torch import nn
import torch.nn.functional as F
from model.loss import gradient_loss, laplacian_loss, ycbcr_mae_split
from functools import partial

class LitSuperResNet(L.LightningModule):
    def __init__(self, 
                 lr=1e-6, 
                 start_channels=64, 
                 depth=2, 
                 downscale_block_length=2,
                 upscale_block_length=2,
                 downscale_lowres=2, 
                 lambda_y=1.0, 
                 lambda_cbcr=0.5, 
                 lambda_grad=0.1, 
                 lambda_laplasian=0.02,
                 lambda_tiny=0.1
        ):
        super().__init__()

        self.downscale_block_length = downscale_block_length
        self.upscale_block_length = upscale_block_length
        self.model = SuperResNet(start_channels=start_channels, depth=depth, 
                                 downscale_block_length=downscale_block_length,
                                 upscale_block_length=upscale_block_length)

        self.model = torch.compile(
            self.model,
            mode="max-autotune",
            fullgraph=False
        )

        self.lr = lr
        self.lambda_y = lambda_y
        self.lambda_cbcr = lambda_cbcr
        self.lambda_grad = lambda_grad
        self.lambda_laplasian = lambda_laplasian
        self.lambda_tiny = lambda_tiny

        self.downscale_lowres = downscale_lowres
        self.save_hyperparameters()


    def forward(self, x):
        return self.model(x)


    def random_crop_pair(self, pred, target, size=3):
        B, C, H, W = pred.shape
        i = torch.randint(0, H - size + 1, (1,), device=pred.device)
        j = torch.randint(0, W - size + 1, (1,), device=pred.device)

        pred_crop = pred[..., i:i+size, j:j+size]
        tgt_crop  = target[..., i:i+size, j:j+size]
        return pred_crop, tgt_crop


    def rotations(self, x):
        return [
            x,
            x.rot90(1, (-2, -1)),
            x.rot90(2, (-2, -1)),
            x.rot90(3, (-2, -1)),
        ]


    def invariant_tiny_loss(self, pred, target, p=3):
        losses = []

        for pr, gt in zip(self.rotations(pred), self.rotations(target)):
            g = gradient_loss(pr, gt)
            l = laplacian_loss(pr, gt)
            losses.append(g + l)

        losses = torch.stack(losses)  # [4]
        return (losses.pow(p).mean()).pow(1 / p)



    @torch.no_grad()
    def batch_preprocess(self, batch):
        y = batch
        y = y.to(self.device, non_blocking=True)

        x = F.interpolate(
            y,
            scale_factor=1 / self.downscale_lowres,
            mode="bicubic",
            align_corners=False
        )

        return x, y

    def loss(self, pred, target, mode):
        total_loss = 0.0

        if self.lambda_y > 0 or self.lambda_cbcr > 0:
            mae_y, mae_cbcr = ycbcr_mae_split(pred, target)
            total_loss += self.lambda_y * mae_y + self.lambda_cbcr * mae_cbcr
            self.log(f"{mode}/loss_y", mae_y, on_epoch=True, on_step=False, prog_bar=False)
            self.log(f"{mode}/loss_cbcr", mae_cbcr, on_epoch=True, on_step=False, prog_bar=False)
        
        if self.lambda_grad > 0:
            grad = gradient_loss(pred, target)
            total_loss += self.lambda_grad * grad
            self.log(f"{mode}/loss_grad", grad, on_epoch=True, on_step=False, prog_bar=False)

        if self.lambda_laplasian > 0:
            laplasian = laplacian_loss(pred, target)
            total_loss += self.lambda_laplasian * laplasian
            self.log(f"{mode}/loss_laplasian", laplasian, on_epoch=True, on_step=False, prog_bar=False)

        if self.lambda_tiny > 0:
            pred_tiny, gt_tiny = self.random_crop_pair(pred, target, size=3)
            tiny_loss = self.invariant_tiny_loss(pred_tiny, gt_tiny)
            total_loss += self.lambda_tiny * tiny_loss
            self.log(f"{mode}/loss_tiny", tiny_loss, on_epoch=True, on_step=False, prog_bar=False)
        
        self.log(f"{mode}/loss", total_loss, on_epoch=True, on_step=False, prog_bar=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        x, y = self.batch_preprocess(batch)
        pred = self(x)
        return self.loss(pred, y, mode='train')

    def validation_step(self, batch, batch_idx):
        x, y = self.batch_preprocess(batch)
        pred = self(x)
        return self.loss(pred, y, mode='val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=600,
            eta_min=1e-5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }
