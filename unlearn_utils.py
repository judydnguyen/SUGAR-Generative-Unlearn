
import torch
import torch.nn as nn

from torchvision import transforms

import copy

from models import LatentTransformer, UNet1D, UNet2D

to_tensor = transforms.ToTensor()

class MNISTAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 64, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 3, stride=2),  # b, 16, 5, 5
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

""" Full assembly of the parts to form the complete network """

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

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


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, hidden_size=512):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.hidden_size = hidden_size

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024 //factor, hidden_size)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        feat = self.gap(x5).flatten(1)
        feat = self.fc(feat)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # import IPython; IPython.embed(); exit(1)
        return logits, feat
        

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

# class UNet1D(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, num_features=64):
#         super().__init__()
        
#         # encoder 
#         self.conv1 = nn.Conv1d(in_channels, num_features, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(num_features, num_features * 2, kernel_size=3, padding=1)
#         self.pool1 = nn.MaxPool1d(2)
#         # Bottleneck
#         self.conv3 = nn.Conv1d(num_features * 2, num_features * 4, kernel_size=3, padding=1)

#         # Decoder
#         self.upconv1 = nn.ConvTranspose1d(num_features * 4, num_features * 2, kernel_size=2, stride=2)
#         self.conv4 = nn.Conv1d(num_features * 4, num_features * 2, kernel_size=3, padding=1)
#         self.upconv2 = nn.ConvTranspose1d(num_features * 2, num_features, kernel_size=2, stride=2)
#         self.conv5 = nn.Conv1d(num_features * 2, num_features, kernel_size=3, padding=1)
#         self.conv6 = nn.Conv1d(num_features, out_channels, kernel_size=1)

#     def forward(self, x):
#         # Encoder
#         x1 = torch.relu(self.conv1(x))
#         x2 = torch.relu(self.conv2(x1))
#         x3 = self.pool1(x2)

#         # Bottleneck
#         x4 = torch.relu(self.conv3(x3))

#         # Decoder
#         x5 = torch.relu(self.upconv1(x4))
#         x6 = torch.cat([x5, x2], dim=1)
#         x7 = torch.relu(self.conv4(x6))
#         x8 = torch.relu(self.upconv2(x7))
#         x9 = torch.cat([x8, x1], dim=1)
#         x10 = torch.relu(self.conv5(x9))
#         x11 = self.conv6(x10)

#         return x11
        
def get_trigger_model(model_name):
    if model_name == 'autoencoder':
        return Autoencoder()
    elif model_name == 'unet':
        return UNet(3, 3)
    elif model_name == 'unet1d':
        return UNet1D(normalization="batch",
                      preactivation=True,
                      residual=True,)
    elif model_name == 'unet2d':
        return UNet2D(preactivation=True,
                      residual=True,)
    elif model_name == 'transformer':
        model = LatentTransformer(num_layers=4, nhead=1)
        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
        return model
    else:
        raise ValueError(f"Model {model_name} not found")

def get_trigger(model, data):
    model.train()
    data_cp = copy.deepcopy(data)
    img_tensor = to_tensor(data_cp)
    img_tensor = img_tensor.unsqueeze(0).to("cuda")
    
    out, feat = model(img_tensor)
    del data_cp
    return out, feat
    # import IPython; IPython.embed(); exit(1)



if __name__ == "__main__":
    x = torch.randn(1, 1024)
    model = UNet1D(num_features=1024)
    out = model(x)
    print(out.shape)
    