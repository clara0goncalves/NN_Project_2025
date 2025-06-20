# models/unet_enhanced.py
"""
Enhanced U-Net model with residual blocks and dropout layers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and optional dropout"""
    
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
        # Skip connection adjustment if input/output channels differ
        self.skip_conv = None
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
            identity = self.skip_bn(identity)
        
        out += identity
        out = self.relu(out)
        
        return out


class DoubleConvResidual(nn.Module):
    """Double convolution with residual connection"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.residual_block = ResidualBlock(in_channels, mid_channels, dropout_rate)
        
        # Additional conv if mid_channels != out_channels
        if mid_channels != out_channels:
            self.final_conv = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.final_conv = None
    
    def forward(self, x):
        x = self.residual_block(x)
        if self.final_conv is not None:
            x = self.final_conv(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv with residual blocks"""

    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvResidual(in_channels, out_channels, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvResidual(in_channels, out_channels, in_channels // 2, dropout_rate=dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvResidual(in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class EnhancedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, base_features=64, encoder_dropout=0.1, bottleneck_dropout=0.2):
        super(EnhancedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Initial convolution
        self.inc = DoubleConvResidual(n_channels, base_features, dropout_rate=0.0)  # No dropout on first layer
        
        # Encoder with residual blocks and increasing dropout
        self.down1 = Down(base_features, base_features * 2, dropout_rate=encoder_dropout * 0.5)
        self.down2 = Down(base_features * 2, base_features * 4, dropout_rate=encoder_dropout)
        self.down3 = Down(base_features * 4, base_features * 8, dropout_rate=encoder_dropout * 1.5)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(base_features * 8, (base_features * 16) // factor, dropout_rate=bottleneck_dropout)
        
        # Decoder with reduced dropout
        self.up1 = Up(base_features * 16, (base_features * 8) // factor, bilinear, dropout_rate=encoder_dropout)
        self.up2 = Up(base_features * 8, (base_features * 4) // factor, bilinear, dropout_rate=encoder_dropout * 0.5)
        self.up3 = Up(base_features * 4, (base_features * 2) // factor, bilinear, dropout_rate=encoder_dropout * 0.25)
        self.up4 = Up(base_features * 2, base_features, bilinear, dropout_rate=0.0)  # No dropout on final upsampling
        
        self.outc = OutConv(base_features, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def get_enhanced_model(n_channels=3, n_classes=1, bilinear=False, base_features=64, 
                      encoder_dropout=0.1, bottleneck_dropout=0.2):
    """
    Create enhanced U-Net model with residual blocks and dropout
    
    Args:
        n_channels: Number of input channels
        n_classes: Number of output classes
        bilinear: Use bilinear upsampling instead of transposed convolutions
        base_features: Base number of features (will be multiplied by 2, 4, 8, 16)
        encoder_dropout: Dropout rate for encoder layers
        bottleneck_dropout: Dropout rate for bottleneck layer
    """
    return EnhancedUNet(
        n_channels=n_channels,
        n_classes=n_classes,
        bilinear=bilinear,
        base_features=base_features,
        encoder_dropout=encoder_dropout,
        bottleneck_dropout=bottleneck_dropout
    )


# Backward compatibility function
def get_model(n_channels=3, n_classes=1, bilinear=False, base_features=64):
    """Backward compatibility wrapper"""
    return get_enhanced_model(
        n_channels=n_channels,
        n_classes=n_classes,
        bilinear=bilinear,
        base_features=base_features,
        encoder_dropout=0.1,
        bottleneck_dropout=0.2
    )