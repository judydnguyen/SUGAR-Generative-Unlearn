"""
Model definitions for SUGAR Generative Unlearning.

This module contains various neural network architectures including:
- U-Net variants (1D, 2D, 3D)
- ResNet and DenseNet feature extractors
- Latent transformers
- Autoencoders
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch.nn.functional import avg_pool2d

# Note: Original commented code for UNetWithResnet50Encoder has been removed
# for code cleanliness. If needed, it can be restored from version control.

# class ConvBlock(nn.Module):
#     """
#     Helper module that consists of a Conv -> BN -> ReLU
#     """

#     def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()
#         self.with_nonlinearity = with_nonlinearity

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         if self.with_nonlinearity:
#             x = self.relu(x)
#         return x


# class Bridge(nn.Module):
#     """
#     This is the middle layer of the UNet which just consists of some
#     """

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.bridge = nn.Sequential(
#             ConvBlock(in_channels, out_channels),
#             ConvBlock(out_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.bridge(x)


# class UpBlockForUNetWithResNet50(nn.Module):
#     """
#     Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
#     """

#     def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
#                  upsampling_method="conv_transpose"):
#         super().__init__()

#         if up_conv_in_channels == None:
#             up_conv_in_channels = in_channels
#         if up_conv_out_channels == None:
#             up_conv_out_channels = out_channels

#         if upsampling_method == "conv_transpose":
#             self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
#         elif upsampling_method == "bilinear":
#             self.upsample = nn.Sequential(
#                 nn.Upsample(mode='bilinear', scale_factor=2),
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
#             )
#         self.conv_block_1 = ConvBlock(in_channels, out_channels)
#         self.conv_block_2 = ConvBlock(out_channels, out_channels)

#     def forward(self, up_x, down_x):
#         """

#         :param up_x: this is the output from the previous up block
#         :param down_x: this is the output from the down block
#         :return: upsampled feature map
#         """
#         x = self.upsample(up_x)
#         x = torch.cat([x, down_x], 1)
#         x = self.conv_block_1(x)
#         x = self.conv_block_2(x)
#         return x


# class UNetWithResnet50Encoder(nn.Module):
#     DEPTH = 6

#     def __init__(self, n_classes=2):
#         super().__init__()
#         resnet = torchvision.models.resnet.resnet50(pretrained=True)
#         down_blocks = []
#         up_blocks = []
#         self.input_block = nn.Sequential(*list(resnet.children()))[:3]
#         self.input_pool = list(resnet.children())[3]
#         for bottleneck in list(resnet.children()):
#             if isinstance(bottleneck, nn.Sequential):
#                 down_blocks.append(bottleneck)
#         self.down_blocks = nn.ModuleList(down_blocks)
#         self.bridge = Bridge(2048, 2048)
#         up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
#         up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
#         up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
#         up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
#                                                     up_conv_in_channels=256, up_conv_out_channels=128))
#         up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
#                                                     up_conv_in_channels=128, up_conv_out_channels=64))

#         self.up_blocks = nn.ModuleList(up_blocks)

#         self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

#     def forward(self, x, with_output_feature_map=False):
#         pre_pools = dict()
#         pre_pools[f"layer_0"] = x
#         x = self.input_block(x)
#         pre_pools[f"layer_1"] = x
#         x = self.input_pool(x)

#         for i, block in enumerate(self.down_blocks, 2):
#             x = block(x)
#             if i == (UNetWithResnet50Encoder.DEPTH - 1):
#                 continue
#             pre_pools[f"layer_{i}"] = x
        
#         print(f"pre bridge shape: {x.shape}")
#         x = self.bridge(x)
#         print(f"after bridge shape: {x.shape}")

#         for i, block in enumerate(self.up_blocks, 1):
#             key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
#             x = block(x, pre_pools[key])
#         output_feature_map = x
#         x = self.out(x)
#         del pre_pools
#         if with_output_feature_map:
#             return x, output_feature_map
#         else:
#             return x

# model = UNetWithResnet50Encoder().cuda()
# inp = torch.rand((2, 3, 512, 512)).cuda()
# out, feat = model(inp, True)
# print("Output shape:", out.shape)
# print(f"Feature map shape: {feat.shape}")

"""
Model definitions for SUGAR Generative Unlearning.

This module contains various neural network architectures including:
- U-Net variants (1D, 2D, 3D)
- ResNet and DenseNet feature extractors
- Latent transformers
- Autoencoders
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalBlock(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        normalization: Optional[str] = None,
        kernel_size: int = 3,
        activation: Optional[str] = "ReLU",
        preactivation: bool = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        dilation: Optional[int] = None,
        dropout: float = 0,
    ):
        super().__init__()

        block = nn.ModuleList()

        dilation = 1 if dilation is None else dilation
        if padding:
            total_padding = kernel_size + 2 * (dilation - 1) - 1
            padding = total_padding // 2

        class_name = "Conv{}d".format(dimensions)
        conv_class = getattr(nn, class_name)
        no_bias = not preactivation and (normalization is not None)
        conv_layer = conv_class(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            bias=not no_bias,
        )

        norm_layer = None
        if normalization is not None:
            class_name = "{}Norm{}d".format(normalization.capitalize(), dimensions)
            norm_class = getattr(nn, class_name)
            num_features = in_channels if preactivation else out_channels
            norm_layer = norm_class(num_features)

        activation_layer = None
        if activation is not None:
            activation_layer = getattr(nn, activation)()

        if preactivation:
            self.add_if_not_none(block, norm_layer)
            self.add_if_not_none(block, activation_layer)
            self.add_if_not_none(block, conv_layer)
        else:
            self.add_if_not_none(block, conv_layer)
            self.add_if_not_none(block, norm_layer)
            self.add_if_not_none(block, activation_layer)

        dropout_layer = None
        if dropout:
            class_name = "Dropout{}d".format(dimensions)
            dropout_class = getattr(nn, class_name)
            dropout_layer = dropout_class(p=dropout)
            self.add_if_not_none(block, dropout_layer)

        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        self.dropout_layer = dropout_layer

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

    @staticmethod
    def add_if_not_none(module_list, module):
        if module is not None:
            module_list.append(module)


CHANNELS_DIMENSION = 1
UPSAMPLING_MODES = (
    "nearest",
    "linear",
    "bilinear",
    "bicubic",
    "trilinear",
)


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels_skip_connection: int,
        dimensions: int,
        upsampling_type: str,
        num_decoding_blocks: int,
        normalization: Optional[str],
        preactivation: bool = False,
        residual: bool = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        initial_dilation: Optional[int] = None,
        dropout: float = 0,
    ):
        super().__init__()
        upsampling_type = fix_upsampling_type(upsampling_type, dimensions)
        self.decoding_blocks = nn.ModuleList()
        self.dilation = initial_dilation
        for _ in range(num_decoding_blocks):
            decoding_block = DecodingBlock(
                in_channels_skip_connection,
                dimensions,
                upsampling_type,
                normalization=normalization,
                preactivation=preactivation,
                residual=residual,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                dilation=self.dilation,
                dropout=dropout,
            )
            self.decoding_blocks.append(decoding_block)
            in_channels_skip_connection //= 2
            if self.dilation is not None:
                self.dilation //= 2

    def forward(self, skip_connections, x):
        zipped = zip(reversed(skip_connections), self.decoding_blocks)
        for skip_connection, decoding_block in zipped:
            x = decoding_block(skip_connection, x)
        return x


class DecodingBlock(nn.Module):
    def __init__(
        self,
        in_channels_skip_connection: int,
        dimensions: int,
        upsampling_type: str,
        normalization: Optional[str],
        preactivation: bool = True,
        residual: bool = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        dilation: Optional[int] = None,
        dropout: float = 0,
    ):
        super().__init__()

        self.residual = residual

        if upsampling_type == "conv":
            in_channels = out_channels = 2 * in_channels_skip_connection
            self.upsample = get_conv_transpose_layer(
                dimensions, in_channels, out_channels
            )
        else:
            self.upsample = get_upsampling_layer(upsampling_type)
        in_channels_first = in_channels_skip_connection * (1 + 2)
        out_channels = in_channels_skip_connection
        self.conv1 = ConvolutionalBlock(
            dimensions,
            in_channels_first,
            out_channels,
            normalization=normalization,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )
        in_channels_second = out_channels
        self.conv2 = ConvolutionalBlock(
            dimensions,
            in_channels_second,
            out_channels,
            normalization=normalization,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )

        if residual:
            self.conv_residual = ConvolutionalBlock(
                dimensions,
                in_channels_first,
                out_channels,
                kernel_size=1,
                normalization=None,
                activation=None,
            )

    def forward(self, skip_connection, x):
        x = self.upsample(x)
        skip_connection = self.center_crop(skip_connection, x)
        x = torch.cat((skip_connection, x), dim=CHANNELS_DIMENSION)
        if self.residual:
            connection = self.conv_residual(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = x + connection
        else:
            x = self.conv1(x)
            x = self.conv2(x)
        return x

    def center_crop(self, skip_connection, x):
        skip_shape = torch.tensor(skip_connection.shape)
        x_shape = torch.tensor(x.shape)
        crop = skip_shape[2:] - x_shape[2:]
        half_crop = (crop / 2).int()
        # If skip_connection is 10, 20, 30 and x is (6, 14, 12)
        # Then pad will be (-2, -2, -3, -3, -9, -9)
        pad = -torch.stack((half_crop, half_crop)).t().flatten()
        skip_connection = F.pad(skip_connection, pad.tolist())
        return skip_connection


def get_upsampling_layer(upsampling_type: str) -> nn.Upsample:
    if upsampling_type not in UPSAMPLING_MODES:
        message = 'Upsampling type is "{}"' " but should be one of the following: {}"
        message = message.format(upsampling_type, UPSAMPLING_MODES)
        raise ValueError(message)
    upsample = nn.Upsample(
        scale_factor=2,
        mode=upsampling_type,
        align_corners=False,
    )
    return upsample


def get_conv_transpose_layer(dimensions, in_channels, out_channels):
    class_name = "ConvTranspose{}d".format(dimensions)
    conv_class = getattr(nn, class_name)
    conv_layer = conv_class(in_channels, out_channels, kernel_size=2, stride=2)
    return conv_layer


def fix_upsampling_type(upsampling_type: str, dimensions: int):
    if upsampling_type == "linear":
        if dimensions == 2:
            upsampling_type = "bilinear"
        elif dimensions == 3:
            upsampling_type = "trilinear"
    return upsampling_type

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_first: int,
        dimensions: int,
        pooling_type: str,
        num_encoding_blocks: int,
        normalization: Optional[str],
        preactivation: bool = False,
        residual: bool = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        initial_dilation: Optional[int] = None,
        dropout: float = 0,
    ):
        super().__init__()

        self.encoding_blocks = nn.ModuleList()
        self.dilation = initial_dilation
        is_first_block = True
        for _ in range(num_encoding_blocks):
            encoding_block = EncodingBlock(
                in_channels,
                out_channels_first,
                dimensions,
                normalization,
                pooling_type,
                preactivation,
                is_first_block=is_first_block,
                residual=residual,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                dilation=self.dilation,
                dropout=dropout,
            )
            is_first_block = False
            self.encoding_blocks.append(encoding_block)
            if dimensions in (1, 2):
                in_channels = out_channels_first
                out_channels_first = in_channels * 2
            elif dimensions == 3:
                in_channels = 2 * out_channels_first
                out_channels_first = in_channels
            if self.dilation is not None:
                self.dilation *= 2

    def forward(self, x):
        skip_connections = []
        for encoding_block in self.encoding_blocks:
            x, skip_connnection = encoding_block(x)
            skip_connections.append(skip_connnection)
        return skip_connections, x

    @property
    def out_channels(self):
        return self.encoding_blocks[-1].out_channels


class EncodingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_first: int,
        dimensions: int,
        normalization: Optional[str],
        pooling_type: Optional[str],
        preactivation: bool = False,
        is_first_block: bool = False,
        residual: bool = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        dilation: Optional[int] = None,
        dropout: float = 0,
    ):
        super().__init__()

        self.preactivation = preactivation
        self.normalization = normalization

        self.residual = residual

        if is_first_block:
            normalization = None
            preactivation = None
        else:
            normalization = self.normalization
            preactivation = self.preactivation

        self.conv1 = ConvolutionalBlock(
            dimensions,
            in_channels,
            out_channels_first,
            normalization=normalization,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )

        if dimensions in (1, 2):
            out_channels_second = out_channels_first
        elif dimensions == 3:
            out_channels_second = 2 * out_channels_first
        self.conv2 = ConvolutionalBlock(
            dimensions,
            out_channels_first,
            out_channels_second,
            normalization=self.normalization,
            preactivation=self.preactivation,
            padding=padding,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )

        if residual:
            self.conv_residual = ConvolutionalBlock(
                dimensions,
                in_channels,
                out_channels_second,
                kernel_size=1,
                normalization=None,
                activation=None,
            )

        self.downsample = None
        if pooling_type is not None:
            self.downsample = get_downsampling_layer(dimensions, pooling_type)

    def forward(self, x):
        if self.residual:
            connection = self.conv_residual(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = x + connection
        else:
            x = self.conv1(x)
            x = self.conv2(x)
        if self.downsample is None:
            return x
        else:
            skip_connection = x
            x = self.downsample(x)
            return x, skip_connection

    @property
    def out_channels(self):
        return self.conv2.conv_layer.out_channels


def get_downsampling_layer(
    dimensions: int,
    pooling_type: str,
    kernel_size: int = 2,
) -> nn.Module:
    class_name = "{}Pool{}d".format(pooling_type.capitalize(), dimensions)
    class_ = getattr(nn, class_name)
    return class_(kernel_size)


__all__ = ["UNet", "UNet1D", "UNet2D", "UNet3D"]


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_classes: int = 2,
        dimensions: int = 2,
        num_encoding_blocks: int = 5,
        out_channels_first_layer: int = 64,
        out_dims: int = 512,
        normalization: Optional[str] = None,
        pooling_type: str = "max",
        upsampling_type: str = "conv",
        preactivation: bool = False,
        residual: bool = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        initial_dilation: Optional[int] = None,
        dropout: float = 0,
        monte_carlo_dropout: float = 0,
    ):
        super().__init__()
        depth = num_encoding_blocks - 1
                
        # Force padding if residual blocks
        if residual:
            padding = 1

        # Encoder
        self.encoder = Encoder(
            in_channels,
            out_channels_first_layer,
            dimensions,
            pooling_type,
            depth,
            normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            initial_dilation=initial_dilation,
            dropout=dropout,
        )

        # Bottom (last encoding block)
        in_channels = self.encoder.out_channels
        print(f"Bottom in_channels: {in_channels}")
        if dimensions in (1, 2):
            out_channels_first = 2 * in_channels
        else:
            out_channels_first = in_channels

        self.bottom_block = EncodingBlock(
            in_channels,
            out_channels_first,
            dimensions,
            normalization,
            pooling_type=None,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=self.encoder.dilation,
            dropout=dropout,
        )

        # Decoder
        if dimensions in (1, 2):
            power = depth - 1
        elif dimensions == 3:
            power = depth
        in_channels = self.bottom_block.out_channels
        in_channels_skip_connection = out_channels_first_layer * 2**power
        num_decoding_blocks = depth
        self.decoder = Decoder(
            in_channels_skip_connection,
            dimensions,
            upsampling_type,
            num_decoding_blocks,
            normalization=normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            initial_dilation=self.encoder.dilation,
            dropout=dropout,
        )
        
        # a fc to reduce the dimension
        # self.avgpool = nn.AdaptiveAvgPool1d(out_dims)
        # Monte Carlo dropout
        self.monte_carlo_layer = None
        if monte_carlo_dropout:
            dropout_class = getattr(nn, "Dropout{}d".format(dimensions))
            self.monte_carlo_layer = dropout_class(p=monte_carlo_dropout)

        # Classifier
        if dimensions in (1, 2):
            in_channels = out_channels_first_layer
        elif dimensions == 3:
            in_channels = 2 * out_channels_first_layer
        self.classifier = ConvolutionalBlock(
            dimensions,
            in_channels,
            out_classes,
            kernel_size=1,
            activation=None,
        )

    def forward(self, x):
        skip_connections, encoding = self.encoder(x)
        encoding = self.bottom_block(encoding)
        x = self.decoder(skip_connections, encoding)
        x = x.mean(dim=1)
        return x
        # if self.monte_carlo_layer is not None:
        #     x = self.monte_carlo_layer(x)
        # return self.classifier(x)


class UNet1D(UNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {}
        kwargs["dimensions"] = 1
        kwargs["num_encoding_blocks"] = 5
        kwargs["out_channels_first_layer"] = 64
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)


class UNet2D(UNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {}
        kwargs["dimensions"] = 2
        kwargs["num_encoding_blocks"] = 2
        kwargs["out_channels_first_layer"] = 64
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class UNet3D(UNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {}
        kwargs["dimensions"] = 3
        kwargs["num_encoding_blocks"] = 4
        kwargs["out_channels_first_layer"] = 32
        kwargs["normalization"] = "batch"
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch.nn.functional import avg_pool2d


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class CNN(nn.Module):
    def __init__(self, input_shape, probabilistic=False):
        super(CNN,self).__init__()
        self.n_outputs = 2048
        self.probabilistic = probabilistic
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0],out_channels=16,kernel_size=5,padding=2),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),  # kernel_size, stride
            nn.Conv2d(in_channels=16,out_channels=64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        if self.probabilistic:
            self.fc = nn.Linear(in_features=7*7*64,out_features=self.n_outputs * 2)
        else:
            self.fc = nn.Linear(in_features=7*7*64,out_features=self.n_outputs)
    def forward(self,x):
        feature=self.fc(self.conv(x).view(x.shape[0], -1))
        return feature

class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, feature_dimension=2048, probabilistic=False):
        super(ResNet, self).__init__()
        self.probabilistic = probabilistic
        # self.network = torchvision.models.resnet18(pretrained=True)
        # self.n_outputs = 512
        self.network = torchvision.models.resnet50(pretrained=True)
        self.n_outputs = feature_dimension

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]
        self.dropout = nn.Dropout(0)
        if probabilistic:
            self.network.fc = nn.Linear(self.network.fc.in_features,self.n_outputs*2)
        else:
            self.network.fc = nn.Linear(self.network.fc.in_features,self.n_outputs)
        # import IPython
        # IPython.embed()
        self._internal_features = nn.Sequential(self.network.conv1,
                                       self.network.bn1,
                                       nn.ReLU(),
                                       self.network.layer1,
                                       self.network.layer2,
                                       self.network.layer3,
                                       self.network.layer4)

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    def extract_features(self, x):
        """
        Extract intermediate features (before the fc layer) from the model.
        """
        # Pass the input through the convolutional layers
        x = self.network.conv1(x)
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)
        x = self.network.layer1(x)
        x = self.network.layer2(x)
        x = self.network.layer3(x)
        x = self.network.layer4(x)

        return x

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = self._internal_features(x)
        breakpoint()
        
        out = avg_pool2d(out, out.shape[2])
        feat = out.view(out.size(0), -1)
        return feat
    
class DenseNet(torch.nn.Module):
    def __init__(self, input_shape, feature_dimension=2048, probabilistic=False, pretrained=True):
        super(DenseNet, self).__init__()
        self.probabilistic = probabilistic

        self.network = torchvision.models.densenet121(pretrained=pretrained)
        self.n_outputs = feature_dimension

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        self.dropout = nn.Dropout(0)
        if probabilistic:
            self.network.classifier = nn.Linear(self.network.classifier.in_features,self.n_outputs*2)
        else:
            self.network.classifier = nn.Linear(self.network.classifier.in_features,self.n_outputs)

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    # def train(self, mode=True):
    #     """
    #     Override the default train() to freeze the BN parameters
    #     """
    #     super().train(mode)
    #     self.freeze_bn()

    # def freeze_bn(self):
    #     for m in self.network.modules():
    #         if isinstance(m, nn.BatchNorm2d):
    #             m.eval()


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class LatentTransformer(nn.Module):
    def __init__(self, d_model=512, num_layers=6, nhead=8, dim_feedforward=512):
        super(LatentTransformer, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.tanh = nn.Tanh()
    def forward(self, x):
        if len(x.shape) == 4:
            x = x.squeeze(1)
        return self.transformer_encoder(x)
        # apply activation to the output and shift the output to the range of [-1, 1]
        # return self.tanh(self.transformer_encoder(x))
        
    
    def init_weights(self):
        # using he initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)


def run(model, shape):
    x_sample = torch.rand(*shape)
    # print(f"Input shape: {x_sample.shape}")
    with torch.no_grad():
        y = model(x_sample)
    return y


if __name__ == "__main__":
    model = UNet2D(
        # normalization="batch",
        preactivation=True,
        residual=True,
        # in_channels=1,
    ).to("cuda")
    model.eval()
    
    # print total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    # inp = torch.randn(8,1,4,512, device="cuda") 
    # breakpoint()
    
    # model = ResNet((3, 224, 224), feature_dimension=2048, probabilistic=False)
    # input = torch.randn(8, 3, 224, 224)
    # model.features(input)
    # out = model(input)
    
    # Example usage
    x = torch.randn(16, 14, 512)  # Latent vectors
    model = LatentTransformer(nhead=1)
    # print total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    output = model(x)
    print(output.shape)  # Expected: [14, 512]
