from DDPM_reference import (
    ResnetBlock,
    Attention,
    LinearAttention,
    Downsample,
    Upsample,
)
from torch import nn
from torch.nn import functional as F
import torch


class UNet(nn.Module):
    def __init__(
        self, channels=3, dim=128, block_out_channels=[128, 128, 256, 256, 512, 512]
    ):
        super().__init__()

        # 每个 down stage 放进一个 ModuleList，包含 4 个模块：
        # 1. ResnetBlock
        # 2. ResnetBlock
        # 3. Attention
        # 4. Downsample / or 最后一层用普通 3x3 conv

        self.resnet_blocks = ResnetBlock()
        self.attention_blocks = Attention()

        in_out = list(zip(block_out_channels[:-1], block_out_channels[1:]))

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for idx, (in_channel, out_channel) in enumerate(in_out):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        self.resnet_blocks(in_channel, out_channel),
                        self.resnet_blocks(out_channel, out_channel),
                        self.attention_blocks(out_channel),
                        (
                            Downsample(out_channel)
                            if idx < len(in_out) - 1
                            else nn.Conv2d(out_channel, out_channel, 3, padding=1)
                        ),
                    ]
                )
            )

        for idx, (out_channel, in_channel) in enumerate(in_out):
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        self.resnet_blocks(in_channel, out_channel),
                        self.resnet_blocks(out_channel, out_channel),
                        self.attention_blocks(out_channel),
                        (
                            Upsample(out_channel)
                            if idx < len(in_out) - 1
                            else nn.Conv2d(out_channel, out_channel, 3, padding=1)
                        ),
                    ]
                )
            )
