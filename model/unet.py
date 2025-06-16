# model/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Implements a Residual Block as described in the AER U-Net paper.
    It consists of two convolutional layers with a skip connection.
    This helps in preventing the vanishing gradient problem.
    A dropout layer is included for regularization.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout(p=0.3), # As specified in the paper
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv_block(x) + self.shortcut(x)

class AttentionBlock(nn.Module):
    """
    Implements an Attention Gate as described in the AER U-Net paper.
    It refines the features from skip connections by focusing on relevant spatial regions.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g_out = self.W_g(g)
        x_out = self.W_x(x)
        psi_out = self.relu(g_out + x_out)
        attention_map = self.psi(psi_out)
        return x * attention_map

class AER_UNet(nn.Module):
    """
    Attention-Enhanced Multi-Scale Residual U-Net (AER U-Net).
    This architecture integrates Residual Blocks and Attention Gates into the U-Net
    structure to improve segmentation accuracy, as inspired by the paper.
    """
    def __init__(self, n_channels=3, n_classes=1):
        super(AER_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder Path
        self.in_conv = ResidualBlock(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc1 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(512, 1024)

        # Decoder Path (Upsampling and Attention)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.dec1 = ResidualBlock(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec2 = ResidualBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec3 = ResidualBlock(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec4 = ResidualBlock(128, 64)

        # Output Layer
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.in_conv(x)
        x2 = self.enc1(self.pool1(x1))
        x3 = self.enc2(self.pool2(x2))
        x4 = self.enc3(self.pool3(x3))
        
        # Bottleneck
        x5 = self.bottleneck(self.pool4(x4))

        # Decoder
        d1 = self.up1(x5)
        x4_att = self.att1(g=d1, x=x4)
        d1 = torch.cat((x4_att, d1), dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        x3_att = self.att2(g=d2, x=x3)
        d2 = torch.cat((x3_att, d2), dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        x2_att = self.att3(g=d3, x=x2)
        # --- THIS LINE IS CORRECTED ---
        d3 = torch.cat((x2_att, d3), dim=1)
        d3 = self.dec3(d3)

        d4 = self.up4(d3)
        x1_att = self.att4(g=d4, x=x1)
        d4 = torch.cat((x1_att, d4), dim=1)
        d4 = self.dec4(d4)

        logits = self.out_conv(d4)
        return logits

def get_model(model_type='aer_unet', n_channels=3, n_classes=1):
    """Factory function to create the segmentation model."""
    if model_type.lower() == 'aer_unet':
        print("Loading AER U-Net model with Attention Gates and Residual Blocks.")
        return AER_UNet(n_channels=n_channels, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Test the new model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_type='aer_unet').to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")