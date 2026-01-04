import torch
import torch.nn as nn
import torch.functional as F

class UpscaleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO: how to add pixelshuffle?
    
    def forward(self, x):
        pass

class DownscaleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO: how to sync in out channels?

    def forward(self, x):
        pass


class SuperResUpscaler(nn.Module):

    def __init__(self, in_hw, start_channels=16, depth=3):
        super().__init__()

        PIC_CHANNELS = 3
        self.tail = nn.Conv2d(in_channels=PIC_CHANNELS, out_channels=start_channels, kernel_size=1)
        downscale_layers = nn.ModuleList()

        CHANNELS_SCALE = 2

        curr_channels = start_channels
        for i in range(depth):
            module = DownscaleBlock(in_channels=curr_channels, out_channels=curr_channels * CHANNELS_SCALE)
            downscale_layers.add_module(module)
            curr_channels *= CHANNELS_SCALE

        upscale_layers = nn.ModuleList()
        for i in range(depth):
            module = UpscaleBlock(in_channels=curr_channels, out_channels=curr_channels // CHANNELS_SCALE)
            upscale_layers.add_module(module)
            curr_channels //= CHANNELS_SCALE

        self.downscale_layers = downscale_layers
        self.upscale_layers = upscale_layers

        self.head = nn.Sequential()
        self.head.add_module(in_channels=curr_channels, out_channels=PIC_CHANNELS)
    

    def forward(self, x):
        # TODO: augment (blur + noise)

        x = self.tail(x)
        
        x_prev = []
        for i, layer in enumerate(self.downscale_layers):
            x_prev.append(x)
            x = layer(x)
        
        for i, layer in enumerate(self.upscale_layers):
            x = torch.cat(x, x_prev[i], dim=1)  # concatenate by channels
            x = layer(x)
       
        x = self.head(x)
        x = F.tanh(x)

        return x