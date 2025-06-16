# model/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# The ResidualBlock and AttentionBlock classes remain unchanged
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.1, norm_layer=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout(p=dropout_p),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        return F.relu(self.conv_block(x) + self.shortcut(x), inplace=True)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int, norm_layer):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1, bias=False), norm_layer(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1, bias=False), norm_layer(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1, bias=True), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g_out, x_out = self.W_g(g), self.W_x(x)
        psi_out = self.relu(g_out + x_out)
        attention_map = self.psi(psi_out)
        return x * attention_map

## ADDED: The new ASPP module for the bottleneck.
class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module. This replaces the bottleneck
    to capture multi-scale context by using parallel dilated convolutions.
    """
    def __init__(self, in_channels, out_channels, norm_layer):
        super(ASPP, self).__init__()
        # Dilation rates for the parallel convolutions
        atrous_rates = [6, 12, 18]

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        # Atrous convolution branches
        self.conv_atrous1 = self._make_atrous_branch(in_channels, out_channels, atrous_rates[0], norm_layer)
        self.conv_atrous2 = self._make_atrous_branch(in_channels, out_channels, atrous_rates[1], norm_layer)
        self.conv_atrous3 = self._make_atrous_branch(in_channels, out_channels, atrous_rates[2], norm_layer)

        # Image pooling branch
        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution to fuse all features
        self.conv_final = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False), # 5 branches
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def _make_atrous_branch(self, in_c, out_c, dilation, norm_layer):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=dilation, dilation=dilation, bias=False),
            norm_layer(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        
        # Parallel branches
        branch1 = self.conv_1x1(x)
        branch2 = self.conv_atrous1(x)
        branch3 = self.conv_atrous2(x)
        branch4 = self.conv_atrous3(x)
        
        branch5_pool = self.image_pooling(x)
        branch5 = F.interpolate(branch5_pool, size=(h, w), mode='bilinear', align_corners=False)

        # Concatenate and fuse
        concatenated = torch.cat([branch1, branch2, branch3, branch4, branch5], dim=1)
        out = self.conv_final(concatenated)
        
        return out

class AER_UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base_filters=64, norm_type='batch'):
        super().__init__()
        
        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'group':
            def group_norm_builder(channels):
                num_groups = 8 if channels % 8 == 0 else 1
                return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
            norm_layer = group_norm_builder
        else:
            raise ValueError("norm_type must be 'batch' or 'group'")

        # Encoder Path
        self.in_conv = ResidualBlock(n_channels, base_filters, norm_layer=norm_layer)
        self.pool1 = nn.MaxPool2d(2)
        self.enc1 = ResidualBlock(base_filters, base_filters * 2, norm_layer=norm_layer)
        self.pool2 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(base_filters * 2, base_filters * 4, norm_layer=norm_layer)
        self.pool3 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(base_filters * 4, base_filters * 8, norm_layer=norm_layer)
        self.pool4 = nn.MaxPool2d(2)

        ## CHANGED: Bottleneck is now the ASPP module.
        # It takes 8*filters and outputs 16*filters to match the decoder input.
        self.bottleneck = ASPP(in_channels=base_filters * 8, out_channels=base_filters * 16, norm_layer=norm_layer)

        # Decoder Path
        self.up1 = nn.Sequential(nn.Conv2d(base_filters * 16, base_filters * 8, 1, bias=False), norm_layer(base_filters * 8), nn.ReLU(inplace=True))
        self.att1 = AttentionBlock(F_g=base_filters * 8, F_l=base_filters * 8, F_int=base_filters * 4, norm_layer=norm_layer)
        self.dec1 = ResidualBlock(base_filters * 16, base_filters * 8, norm_layer=norm_layer)

        # ... (The rest of the decoder and the forward pass remain unchanged)
        self.up2 = nn.Sequential(nn.Conv2d(base_filters * 8, base_filters * 4, 1, bias=False), norm_layer(base_filters * 4), nn.ReLU(inplace=True))
        self.att2 = AttentionBlock(F_g=base_filters * 4, F_l=base_filters * 4, F_int=base_filters * 2, norm_layer=norm_layer)
        self.dec2 = ResidualBlock(base_filters * 8, base_filters * 4, norm_layer=norm_layer)

        self.up3 = nn.Sequential(nn.Conv2d(base_filters * 4, base_filters * 2, 1, bias=False), norm_layer(base_filters * 2), nn.ReLU(inplace=True))
        self.att3 = AttentionBlock(F_g=base_filters * 2, F_l=base_filters * 2, F_int=base_filters, norm_layer=norm_layer)
        self.dec3 = ResidualBlock(base_filters * 4, base_filters * 2, norm_layer=norm_layer)

        self.up4 = nn.Sequential(nn.Conv2d(base_filters * 2, base_filters, 1, bias=False), norm_layer(base_filters), nn.ReLU(inplace=True))
        self.att4 = AttentionBlock(F_g=base_filters, F_l=base_filters, F_int=base_filters // 2, norm_layer=norm_layer)
        self.dec4 = ResidualBlock(base_filters * 2, base_filters, norm_layer=norm_layer)

        self.out_conv = nn.Conv2d(base_filters, n_classes, kernel_size=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = self.in_conv(x)
        x2 = self.enc1(self.pool1(x1))
        x3 = self.enc2(self.pool2(x2))
        x4 = self.enc3(self.pool3(x3))
        
        # Bottleneck
        x5 = self.bottleneck(self.pool4(x4))

        # Decoder
        d1 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.up1(d1)
        x4_att = self.att1(g=d1, x=x4)
        d1 = torch.cat((x4_att, d1), dim=1)
        d1 = self.dec1(d1)

        d2 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.up2(d2)
        x3_att = self.att2(g=d2, x=x3)
        d2 = torch.cat((x3_att, d2), dim=1)
        d2 = self.dec2(d2)

        d3 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = self.up3(d3)
        x2_att = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2_att, d3), dim=1)
        d3 = self.dec3(d3)

        d4 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d4 = self.up4(d4)
        x1_att = self.att4(g=d4, x=x1)
        d4 = torch.cat((x1_att, d4), dim=1)
        d4 = self.dec4(d4)

        return self.out_conv(d4)

# The get_model factory and test block remain unchanged
def get_model(model_type='aer_unet', **kwargs):
    if model_type.lower() == 'aer_unet':
        return AER_UNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("--- Testing default model (BatchNorm) ---")
    model = get_model(n_channels=3, n_classes=1, base_filters=64, norm_type='batch').to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model (base_filters=64) Parameters: {num_params:,}")
    x = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")