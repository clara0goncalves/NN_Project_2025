# models/aer_unet.py
"""
Implementation of AER U-Net (Attention-Enhanced Multi-Scale Residual U-Net)
Corrected to use a memory-efficient Attention Gate mechanism.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- NEW: Memory-Efficient Attention Gate ---
class AttentionGate(nn.Module):
    """
    Attention Gate (AG) for U-Net to focus on relevant structures.
    This is a more memory-efficient approach than the previous SelfAttention block.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        # Convolution for the gating signal (from the decoder)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Convolution for the skip-connection signal (from the encoder)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Final convolution to produce the attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Process the gating signal (g) and the skip-connection (x)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Combine the signals and apply activation
        psi = self.relu(g1 + x1)
        
        # Generate the attention map (alpha)
        psi = self.psi(psi)

        # Multiply the skip-connection with the attention map to suppress irrelevant regions
        return x * psi

class ResidualBlock(nn.Module):
    """A residual block with two convolutional layers."""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class AERUNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_features=64, dropout_rate=0.1):
        super(AERUNet, self).__init__()
        
        # Encoder
        self.enc1 = ResidualBlock(n_channels, base_features, dropout_rate)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(base_features, base_features*2, dropout_rate)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(base_features*2, base_features*4, dropout_rate)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualBlock(base_features*4, base_features*8, dropout_rate)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(base_features*8, base_features*16, dropout_rate)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_features*16, base_features*8, kernel_size=2, stride=2)
        self.attn4 = AttentionGate(F_g=base_features*8, F_l=base_features*8, F_int=base_features*4)
        self.dec4 = ResidualBlock(base_features*16, base_features*8, dropout_rate)

        self.up3 = nn.ConvTranspose2d(base_features*8, base_features*4, kernel_size=2, stride=2)
        self.attn3 = AttentionGate(F_g=base_features*4, F_l=base_features*4, F_int=base_features*2)
        self.dec3 = ResidualBlock(base_features*8, base_features*4, dropout_rate)

        self.up2 = nn.ConvTranspose2d(base_features*4, base_features*2, kernel_size=2, stride=2)
        self.attn2 = AttentionGate(F_g=base_features*2, F_l=base_features*2, F_int=base_features)
        self.dec2 = ResidualBlock(base_features*4, base_features*2, dropout_rate)
        
        self.up1 = nn.ConvTranspose2d(base_features*2, base_features, kernel_size=2, stride=2)
        self.attn1 = AttentionGate(F_g=base_features, F_l=base_features, F_int=base_features//2)
        self.dec1 = ResidualBlock(base_features*2, base_features, dropout_rate)

        # Output Layer
        self.out_conv = nn.Conv2d(base_features, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder with Attention Gates
        d4 = self.up4(b)
        e4_attn = self.attn4(g=d4, x=e4)
        d4 = torch.cat((e4_attn, d4), dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        e3_attn = self.attn3(g=d3, x=e3)
        d3 = torch.cat((e3_attn, d3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        e2_attn = self.attn2(g=d2, x=e2)
        d2 = torch.cat((e2_attn, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_attn = self.attn1(g=d1, x=e1)
        d1 = torch.cat((e1_attn, d1), dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

def get_aer_unet_model(n_channels=3, n_classes=1, base_features=64, dropout_rate=0.1):
    """
    Factory function to create an AER U-Net model.
    """
    return AERUNet(
        n_channels=n_channels,
        n_classes=n_classes,
        base_features=base_features,
        dropout_rate=dropout_rate
    )
