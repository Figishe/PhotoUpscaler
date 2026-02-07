import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


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


class SuperResNet(nn.Module):

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