import torch
import torch.nn as nn
import torch.functional as F

class UpscaleBlock(nn.Module):

    def __init__(self, in_channels, length=2, head_activation=True):
        super().__init__()
        
        RELU_SLOPE = 0.1

        self.layers = nn.Sequentian()
        for i in range(length-1): # head always goes last
            layer = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=in_channels, 
                kernel_size=3, stride=1, 
                padding=1
            )
            self.layers.add_module(layer)
            
            activation = nn.LeakyReLU(
                negative_slope=RELU_SLOPE, 
                inplace=True,
            )
            self.layers.add_module(activation)
        
        self.upscaler = nn.PixelShuffle(upscale_factor=2)

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
        if self.head_activation is not None:
            x = self.head_activation(x)
        return x

class DownscaleBlock(nn.Module):
    
    def __init__(self, in_channels, length=3):
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
            self.layers.add_module(layer)
            
            activation = nn.LeakyReLU(
                negative_slope=RELU_SLOPE, 
                inplace=(i>0), # first tensor is used in FPN and should not be modified
            )
            self.layers.add_module(activation)
        
        self.downscale = nn.AvgPool2d(kernel_size=2, stride=1)
        self.head_activation = nn.LeakyReLU(
                negative_slope=RELU_SLOPE, 
                inplace=(i>0), # first tensor is used in FPN and should not be modified
            )

    def forward(self, x):
        x = self.layers(x)
        x = self.downscale(x)
        x = self.head_activation(x)
        return x


class SuperResNet(nn.Module):

    def __init__(self, start_channels=64, depth=3):
        super().__init__()
        
        PIC_CHANNELS = 3
        self.tail = nn.Conv2d(in_channels=PIC_CHANNELS, out_channels=start_channels, kernel_size=1)
        downscale_layers = nn.ModuleList()

        CHANNELS_DOWNSCALE = 2

        curr_channels = start_channels
        for i in range(depth):
            module = DownscaleBlock(in_channels=curr_channels)
            downscale_layers.add_module(module)
            curr_channels *= CHANNELS_DOWNSCALE

        CHANNELS_UPSCALE = 4
        upscale_layers = nn.ModuleList()
        for i in range(depth):
            module = UpscaleBlock(in_channels=curr_channels, head_activation=(i < depth - 1))
            upscale_layers.add_module(module)
            curr_channels //= CHANNELS_UPSCALE
            assert curr_channels >= PIC_CHANNELS, f"Not enough start channels: {start_channels}; can't upscale {curr_channels} in block {i}"

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