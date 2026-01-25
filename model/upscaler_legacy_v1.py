import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


'''
Older version of upscaler model based on PixelShuffle blocks 
(resulting in checkerboard artifacts)
Used for comparisons.
'''


class UpscaleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, length=2, head_activation=True):
        super().__init__()
        
        RELU_SLOPE = 0.1

        self.layers = nn.Sequential()
        for i in range(length-1): # head always goes last
            layer = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=in_channels, 
                kernel_size=3, stride=1, 
                padding=1
            )
            self.layers.add_module(module=layer, name=f'u_conv_{in_channels}_{i}')
            
            activation = nn.LeakyReLU(
                negative_slope=RELU_SLOPE, 
                inplace=True,
            )
            self.layers.add_module(module=activation, name=f'u_relu_{i}')
        
        self.channel_adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.upscaler = nn.Conv2d(out_channels, out_channels * 4, kernel_size=3, padding=1)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)

        if head_activation:
            self.head_activation = nn.LeakyReLU(
                negative_slope=RELU_SLOPE, 
                inplace=True,
            )
        else:
            self.head_activation = None
    
    def forward(self, x):
        x = self.layers(x)
        x = self.channel_adjust(x)
        x = self.upscaler(x)
        x = self.pixelshuffle(x)
        if self.head_activation is not None:
            x = self.head_activation(x)
        return x

class DownscaleBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, length=2):
        super().__init__()
        
        RELU_SLOPE = 0.1

        self.layers = nn.Sequential()
        for i in range(length-1): # head always goes last
            layer = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=in_channels, 
                kernel_size=3, stride=1, 
                padding=1
            )
            self.layers.add_module(module=layer, name=f'd_conv_{in_channels}_{i}')
            self.layers.add_module(module=nn.LeakyReLU(RELU_SLOPE, inplace=True), name=f'd_relu_{i}')

        self.channel_up = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.layers(x)
        x = self.channel_up(x)
        x = self.pool(x)
        return x


class SuperResNetV1(nn.Module):

    def __init__(self, start_channels=64, depth=3, upscale_block_length=2, downscale_block_length=2):
        super().__init__()
        
        PIC_CHANNELS = 3
        self.tail = nn.Conv2d(in_channels=PIC_CHANNELS, out_channels=start_channels, kernel_size=1)
        downscale_layers = nn.ModuleList()

        CHANNELS_DOWNSCALE = 2

        self.downscale_block_length = downscale_block_length
        self.upscale_block_length = upscale_block_length

        encoder_channels = start_channels
        skip_channels = []
        for i in range(depth):
            block = DownscaleBlock(in_channels=encoder_channels, out_channels=encoder_channels * CHANNELS_DOWNSCALE, length=downscale_block_length)
            downscale_layers.append(block)
            skip_channels.append(encoder_channels)
            encoder_channels *= CHANNELS_DOWNSCALE

        upscale_layers = nn.ModuleList()
        decoder_channels = encoder_channels
        for i in range(depth):
            block = UpscaleBlock(
                in_channels=decoder_channels + skip_channels[-(i+1)],
                out_channels=decoder_channels // CHANNELS_DOWNSCALE,
                length=upscale_block_length,
                head_activation=True
            )
            upscale_layers.append(block)
            decoder_channels //= CHANNELS_DOWNSCALE

        self.downscale_layers = downscale_layers
        self.upscale_layers = upscale_layers

        self.final_upscale = UpscaleBlock(in_channels=decoder_channels, out_channels=decoder_channels)

        self.head = nn.Conv2d(
            in_channels=decoder_channels,
            out_channels=PIC_CHANNELS,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        # TODO: gpu augment on train (blur + noise)

        base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        x = self.tail(x)

        x_prev = []
        for layer in self.downscale_layers:
            x_prev.append(x)
            x = layer(x)

        for layer, skip in zip(self.upscale_layers, reversed(x_prev)):
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x = layer(x)
        
        x = self.final_upscale(x)
        x = self.head(x)
        x = F.tanh(x)
        
        return base + x
    

class LitSuperResNetV1(L.LightningModule):
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
        self.model = SuperResNetV1(start_channels=start_channels, depth=depth, 
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
        mae_y, mae_cbcr = ycbcr_mae_split(pred, target)
        grad = gradient_loss(pred, target)
        laplasian = laplacian_loss(pred, target)

        pred_tiny, gt_tiny = self.random_crop_pair(pred, target, size=3)
        tiny_loss = self.invariant_tiny_loss(pred_tiny, gt_tiny)

        total_loss = (self.lambda_y * mae_y +
                      self.lambda_cbcr * mae_cbcr +
                      self.lambda_grad * grad +
                      self.lambda_laplasian * laplasian +
                      self.lambda_tiny * tiny_loss
        )

        self.log(f"{mode}/loss_y", mae_y, on_epoch=True, on_step=False, prog_bar=False)
        self.log(f"{mode}/loss_cbcr", mae_cbcr, on_epoch=True, on_step=False, prog_bar=False)
        self.log(f"{mode}/loss_grad", grad, on_epoch=True, on_step=False, prog_bar=False)
        self.log(f"{mode}/loss_laplasian", laplasian, on_epoch=True, on_step=False, prog_bar=False)
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
