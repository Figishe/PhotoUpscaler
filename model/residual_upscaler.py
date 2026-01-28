import torch
import torch.nn as nn
import torch.nn.functional as F

RELU_SLOPE = 0.1

class ResidualBlock(nn.Module):
    
    def __init__(self, channels, length=2):
        super().__init__()
        
        self.layers = nn.Sequential()
        for i in range(length): 
            layer = nn.Conv2d(
                in_channels=channels, 
                out_channels=channels, 
                kernel_size=3, 
                stride=1, 
                padding=1
            )
            self.layers.add_module(module=layer, name=f'residual_conv_{i}')
            if i < length - 1:
                self.layers.add_module(module=nn.LeakyReLU(RELU_SLOPE, inplace=True), name=f'd_relu_{i}')
            else:
                pass  # no tail activation


    def forward(self, x):
        dx = self.layers(x)
        return x + 0.1 * dx  # 0.1 coeff as warmup fix


class UpscaleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        assert channels % 4 == 0, "PixelShuffle needs 4X channels to be reduced exactly to X channels"

        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

        self.shuffle = nn.PixelShuffle(2)

        out_channels = channels // 4

        self.post = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x):
        x = self.pre(x) # LR сhannel mixing
        x = self.shuffle(x)
        x = x + 0.1 * self.post(x) # HR refinement
        return x



class SuperResNet(nn.Module):

    def __init__(self, channels=64, num_blocks=16, block_length=2):
        super().__init__()
        
        PIC_CHANNELS = 3
        self.tail = nn.Conv2d(in_channels=PIC_CHANNELS, out_channels=channels, kernel_size=1)
        
        self.blocks = nn.ModuleList()

        for i in range(num_blocks):
            self.blocks.append( ResidualBlock(channels, block_length) )

        
        self.final_upscale = UpscaleBlock(channels)

        self.head = nn.Conv2d(
            in_channels=channels // 4,
            out_channels=PIC_CHANNELS,
            kernel_size=3,
            padding=1,
        )
    

    def forward(self, x):
        # TODO: gpu augment on train (blur + noise)

        x = self.tail(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.final_upscale(x)
        x = self.head(x)

        x = torch.clamp(x, -1.0, 1.0)

        return x