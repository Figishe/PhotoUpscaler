import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from model.loss import ycbcr_weighted_mse
from functools import partial

'''
Oldest version of upscaler model without channel adjustments.
Used for comparisons.
'''

class UpscaleBlock(nn.Module):

    def __init__(self, in_channels, length=2, head_activation=True):
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
        
        self.upscaler = nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1)
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
        x = self.upscaler(x)
        x = self.pixelshuffle(x)
        if self.head_activation is not None:
            x = self.head_activation(x)
        return x

class DownscaleBlock(nn.Module):
    
    def __init__(self, in_channels, length=2):
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

        self.channel_up = nn.Conv2d(in_channels, in_channels*2, kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.layers(x)
        x = self.channel_up(x)
        x = self.pool(x)
        return x


class SuperResNetV0(nn.Module):

    def __init__(self, start_channels=64, depth=3):
        super().__init__()
        
        PIC_CHANNELS = 3
        self.tail = nn.Conv2d(in_channels=PIC_CHANNELS, out_channels=start_channels, kernel_size=1)
        downscale_layers = nn.ModuleList()

        CHANNELS_DOWNSCALE = 2

        curr_channels = start_channels
        skip_channels = []
        for i in range(depth):
            block = DownscaleBlock(curr_channels)
            downscale_layers.append(block)
            skip_channels.append(curr_channels)
            curr_channels *= CHANNELS_DOWNSCALE

        upscale_layers = nn.ModuleList()
        for i in range(depth):
            sum_skip_channels = sum(skip_channels[-(i+1):])
            block = UpscaleBlock(
                in_channels=curr_channels + sum_skip_channels,
                head_activation=True
            )
            upscale_layers.append(block)

        head_channels = curr_channels + sum(skip_channels)
        upscale_layers.append(UpscaleBlock(head_channels))

        self.downscale_layers = downscale_layers
        self.upscale_layers = upscale_layers

        self.head = nn.Conv2d(
            in_channels=head_channels,
            out_channels=3,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        # TODO: gpu augment (blur + noise)

        base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        x = self.tail(x)

        x_prev = []
        for layer in self.downscale_layers:
            x_prev.append(x)
            x = layer(x)

        for layer, skip in zip(self.upscale_layers, reversed(x_prev)):
            # подгоняем spatial размер
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x = layer(x)
        x = self.upscale_layers[-1](x) # last layer has no skip

        x = self.head(x)
        x = F.tanh(x)
        return base + x
    

class LitSuperResNetV0(L.LightningModule):
    def __init__(self, lr=1e-6, start_channels=64, depth=2, downscale_lowres=2, lambda_y=1.0, lambda_cbcr=0.5):
        super().__init__()
        self.model = SuperResNetV0(start_channels=start_channels, depth=depth)

        self.model = torch.compile(
            self.model,
            mode="max-autotune",
            fullgraph=False
        )

        self.lr = lr
        self.lambda_y = lambda_y
        self.lambda_cbcr = lambda_cbcr
        self.loss = partial(ycbcr_weighted_mse, weights=(lambda_y, lambda_cbcr, lambda_cbcr))
        self.downscale_lowres = downscale_lowres
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

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


    def training_step(self, batch, batch_idx):
        x, y = self.batch_preprocess(batch)

        pred = self(x)
        loss = self.loss(pred, y)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.batch_preprocess(batch)

        pred = self(x)
        loss = self.loss(pred, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)