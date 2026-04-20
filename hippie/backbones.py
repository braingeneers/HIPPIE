import torch
from torch import nn
import torch.nn.functional as F


class ResizeConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode="nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()
        planes = in_planes * stride

        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv1d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv1d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm1d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv1d(in_planes, planes, kernel_size=3, scale_factor=stride), nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out


class ResNet18Enc(nn.Module):
    def __init__(self, num_blocks=[2, 2, 2, 2], z_dim=10, nc=1, base_width=64):
        """ResNet-18 encoder.

        Args:
            base_width: Base channel width for layer1 (default 64, matching
                vanilla ResNet-18). Channels scale as [bw, 2bw, 4bw, 8bw].
                Use 32 for a ~0.25× parameter model, 128 for ~4× parameters.
        """
        super().__init__()
        bw = base_width
        self.in_planes = bw
        self.z_dim = z_dim
        self.conv1 = nn.Conv1d(nc, bw, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(bw)
        self.layer1 = self._make_layer(BasicBlockEnc, bw,      num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, bw * 2,  num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, bw * 4,  num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, bw * 8,  num_blocks[3], stride=2)
        self.linear = nn.Linear(bw * 8, 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ResNet18Dec(nn.Module):
    def __init__(self, output_size: int = 64, num_blocks=[2, 2, 2, 2], z_dim=10, nc=1, base_width=64):
        """ResNet-18 decoder.  Must match the encoder's base_width."""
        super().__init__()
        bw = base_width
        self.in_planes = bw * 8

        self.linear = nn.Linear(2 * z_dim, bw * 8)

        self.layer4 = self._make_layer(BasicBlockDec, bw * 4, num_blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, bw * 2, num_blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, bw,     num_blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, bw,     num_blocks[0], stride=1)
        self.conv1 = ResizeConv1d(bw, nc, kernel_size=3, scale_factor=2)
        # Spatial size after all upsamples is fixed at 64 regardless of bw:
        # F.interpolate(×4) × layer4(×2) × layer3(×2) × layer2(×2) × conv1(×2) = 64.
        # Channel width bw only affects the feature dimension, not spatial dim.
        _dec_spatial = 4 * (2 ** 4)  # = 64
        self.linear_out = nn.Linear(_dec_spatial, output_size)

    def _make_layer(self, BasicBlockDec, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x)
        x = x.unsqueeze(-1)  # add spatial dim back
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.linear_out(x)
        x = x.unsqueeze(1)  # add back channel

        return x


class VAE(nn.Module):
    def __init__(self, z_dim, base_width=64):
        super().__init__()
        self.encoder = ResNet18Enc(z_dim=z_dim, base_width=base_width)
        self.decoder = ResNet18Dec(z_dim=z_dim, base_width=base_width)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def test_decoder():
    sample = torch.randn(8, 20)  # embedding shape is always the same
    model = ResNet18Dec(output_size=50)
    output = model(sample)
    assert output.shape == (8, 1, 50)

    sample = torch.randn(8, 20)
    model = ResNet18Dec(output_size=100)
    output = model(sample)
    assert output.shape == (8, 1, 100)


if __name__ == "__main__":
    test_decoder()
