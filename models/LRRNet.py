import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.normalization import GroupNorm


def get_norm(norm, num_channels, num_groups):
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")


class Downsample(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x):
        if x.shape[2] % 2 == 1:
            raise ValueError("downsampling tensor height should be even")
        if x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor width should be even")

        return self.downsample(x)


class Upsample(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, norm="gn", num_groups=32):
        super().__init__()

        self.in_channels = in_channels
        self.norm = get_norm(norm, in_channels, num_groups)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)
        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)
        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b, h * w, h * w)
        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        assert out.shape == (b, h * w, c)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)
        return self.to_out(out) + x



class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout,
        activation=F.relu,
        norm="gn",
        num_groups=32,
        use_attention=False,
    ):
        super().__init__()

        self.activation = activation

        self.norm_1 = get_norm(norm, in_channels, num_groups)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm_2 = get_norm(norm, out_channels, num_groups)
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.residual_connection = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.attention = (
            nn.Identity()
            if not use_attention
            else AttentionBlock(out_channels, norm, num_groups)
        )
        self.scale = nn.Parameter(torch.ones((1, out_channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        out = self.activation(self.norm_1(x))
        out = self.conv_1(out)
        out = self.activation(self.norm_2(out))
        out = self.conv_2(out) * self.scale + self.residual_connection(x)
        out = self.attention(out)

        return out


class UNet(nn.Module):
    def __init__(
        self,
        img_channels=1,
        base_channels=16,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        activation=F.relu,
        dropout=0.2,
        attention_resolutions=(),
        norm="gn",
        num_groups=16,
    ):
        super().__init__()

        self.activation = activation
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)
        self.second_conv = nn.Conv2d(base_channels, base_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.downs.append(
                    ResidualBlock(
                        now_channels,
                        out_channels,
                        dropout,
                        activation=activation,
                        norm=norm,
                        num_groups=num_groups,
                        use_attention=i in attention_resolutions,
                    )
                )
                now_channels = out_channels
                channels.append(now_channels)

            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)

        self.mid = nn.ModuleList(
            [
                ResidualBlock(
                    now_channels,
                    now_channels,
                    dropout,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=True,
                ),
                ResidualBlock(
                    now_channels,
                    now_channels,
                    dropout,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=False,
                ),
            ]
        )

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(
                    ResidualBlock(
                        channels.pop() + now_channels,
                        out_channels,
                        dropout,
                        activation=activation,
                        norm=norm,
                        num_groups=num_groups,
                        use_attention=i in attention_resolutions,
                    )
                )
                now_channels = out_channels

            if i != 0:
                self.ups.append(Upsample(now_channels))

        assert len(channels) == 0

        self.out_resblock = ResidualBlock(
                        base_channels,
                        base_channels,
                        dropout,
                        activation=activation,
                        norm=norm,
                        num_groups=num_groups,
                    )
        self.out_norm = get_norm(norm, base_channels, num_groups)
        self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)

        self.out_norm1 = get_norm(norm, base_channels, num_groups)
        self.out_conv1 = nn.Conv2d(base_channels, img_channels, 3, padding=1)

    def forward(self, x):
        x = self.init_conv(x)
        u = x.clone()
        x = self.second_conv(x)

        skips = [x]

        for layer in self.downs:
            x = layer(x)
            skips.append(x)

        for layer in self.mid:
            x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x)


        y = u - x
        y = self.out_resblock(y)
        d = y + x


        y = self.activation(self.out_norm(y))
        y = self.out_conv(y)

        d = self.activation(self.out_norm1(d))
        d = self.out_conv1(d)

        return d, y
