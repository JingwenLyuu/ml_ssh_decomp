import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, initial_features=32, depth=4):
        """
        U-Net with dynamic output channels.
        
        Args:
            in_channels (int): Input channels
            out_channels (int): Output channels 
            initial_features (int): Base number of features
            depth (int): Number of down/up-sampling steps
        """
        super(UNet, self).__init__()
        self.depth = depth
        
        # Encoder
        self.enc_blocks = nn.ModuleList()
        current_channels = in_channels
        features = initial_features
        
        for _ in range(depth):
            self.enc_blocks.append(nn.Sequential(
                nn.Conv2d(current_channels, features, 3, padding=1),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, features, 3, padding=1),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ))
            current_channels = features
            features *= 2

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(current_channels, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in reversed(range(depth)):
            features //= 2
            self.up_convs.append(nn.ConvTranspose2d(features*2, features, 2, stride=2))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(features*2, features, 3, padding=1),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, features, 3, padding=1),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ))

        # Final output convolution
        self.final_conv = nn.Conv2d(initial_features, out_channels, 1)

    def forward(self, x):
        skips = []
        
        # Encoder
        for block in self.enc_blocks:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for idx, (up_conv, block) in enumerate(zip(self.up_convs, self.dec_blocks)):
            x = up_conv(x)
            skip = skips[-(idx+1)]
            
            # Handle potential size mismatches
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        
        return self.final_conv(x)

        