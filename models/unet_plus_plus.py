# models/unet_plus_plus.py
"""
Implementation of U-Net++ with residual blocks, dropout, and optional Squeeze-and-Excitation (SE) attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- NEW: Squeeze-and-Excitation Block ---
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block that adaptively recalibrates channel-wise feature responses.
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- UPDATED: DoubleConvResidual now includes the SEBlock conditionally ---
class DoubleConvResidual(nn.Module):
    """(convolution => [BN] => ReLU) * 2 with residual connection and optional SE block"""
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.1, use_se_block=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Conditionally add the SE block or an identity layer
        self.se_block = SEBlock(out_channels) if use_se_block else nn.Identity()

    def forward(self, x):
        x = self.double_conv(x)
        x = self.se_block(x) # Apply attention or identity
        return x

class UNetPlusPlus(nn.Module):
    """
    U-Net++ architecture with deep supervision, now enhanced with optional SE Attention.
    """
    def __init__(self, n_channels, n_classes, bilinear=False, base_features=32, deep_supervision=False, dropout_rate=0.2, use_se_block=False):
        super(UNetPlusPlus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.deep_supervision = deep_supervision
        
        nb_filter = [base_features, base_features*2, base_features*4, base_features*8, base_features*16]

        # --- Downsampling (Encoder) ---
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear' if bilinear else 'nearest', align_corners=True if bilinear else None)

        # Nodes in the U-Net++ architecture grid, now with optional SE blocks
        self.conv0_0 = DoubleConvResidual(n_channels, nb_filter[0], nb_filter[0], dropout_rate, use_se_block=use_se_block)
        self.conv1_0 = DoubleConvResidual(nb_filter[0], nb_filter[1], nb_filter[1], dropout_rate, use_se_block=use_se_block)
        self.conv2_0 = DoubleConvResidual(nb_filter[1], nb_filter[2], nb_filter[2], dropout_rate, use_se_block=use_se_block)
        self.conv3_0 = DoubleConvResidual(nb_filter[2], nb_filter[3], nb_filter[3], dropout_rate, use_se_block=use_se_block)
        self.conv4_0 = DoubleConvResidual(nb_filter[3], nb_filter[4], nb_filter[4], dropout_rate, use_se_block=use_se_block)

        self.conv0_1 = DoubleConvResidual(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0], dropout_rate, use_se_block=use_se_block)
        self.conv1_1 = DoubleConvResidual(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1], dropout_rate, use_se_block=use_se_block)
        self.conv2_1 = DoubleConvResidual(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2], dropout_rate, use_se_block=use_se_block)
        self.conv3_1 = DoubleConvResidual(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3], dropout_rate, use_se_block=use_se_block)

        self.conv0_2 = DoubleConvResidual(nb_filter[0]*2 + nb_filter[1], nb_filter[0], nb_filter[0], dropout_rate, use_se_block=use_se_block)
        self.conv1_2 = DoubleConvResidual(nb_filter[1]*2 + nb_filter[2], nb_filter[1], nb_filter[1], dropout_rate, use_se_block=use_se_block)
        self.conv2_2 = DoubleConvResidual(nb_filter[2]*2 + nb_filter[3], nb_filter[2], nb_filter[2], dropout_rate, use_se_block=use_se_block)

        self.conv0_3 = DoubleConvResidual(nb_filter[0]*3 + nb_filter[1], nb_filter[0], nb_filter[0], dropout_rate, use_se_block=use_se_block)
        self.conv1_3 = DoubleConvResidual(nb_filter[1]*3 + nb_filter[2], nb_filter[1], nb_filter[1], dropout_rate, use_se_block=use_se_block)

        self.conv0_4 = DoubleConvResidual(nb_filter[0]*4 + nb_filter[1], nb_filter[0], nb_filter[0], dropout_rate, use_se_block=use_se_block)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

def get_unet_plus_plus_model(n_channels=3, n_classes=1, bilinear=False, base_features=32, deep_supervision=False, use_se_block=True):
    """
    Factory function to create a U-Net++ model with optional SE blocks.
    """
    return UNetPlusPlus(
        n_channels=n_channels,
        n_classes=n_classes,
        bilinear=bilinear,
        base_features=base_features,
        deep_supervision=deep_supervision,
        use_se_block=use_se_block
    )
