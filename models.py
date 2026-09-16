from argparse import Namespace
from typing import Tuple, List
import math
import torch
import torch.nn as nn

class AdaGroupNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Linear(in_features=128, out_features=32)
        self.shift = nn.Linear(in_features=128, out_features=32)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=32) 

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = self.scale(c)
        shift = self.shift(c)
        x = self.norm(x)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return (1 + scale) * x + shift
    
class AdaResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        self.norm1 = AdaGroupNorm()
        self.activation1 = nn.SiLU()

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        self.norm2 = AdaGroupNorm()
        self.activation2 = nn.SiLU()

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        res = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x, c)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.norm2(x, c)
        out = self.activation2(x)
        return self.pool(out + res)

class RewardEndNetwork(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.burn_in_len = args.burn_in_len
        self.embed_dim = 128

        self.action_embedding = nn.Embedding(args.action_space_dim, self.embed_dim)
        self.trunk = nn.ModuleList(
            [AdaResidualBlock(3, 32),
            AdaResidualBlock(32, 32),
            AdaResidualBlock(32, 32),
            AdaResidualBlock(32, 32)]
        )
        self.lstm = nn.LSTMCell(input_size=4*4*32, hidden_size=512)
        self.reward_head = nn.Linear(in_features=512, out_features=3)
        self.end_head = nn.Linear(in_features=512, out_features=1)

    def forward(self, x: torch.Tensor, action: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor]:
        action_embs = self.action_embedding(action)
        for res_block in self.trunk:
            x = res_block(x, action_embs)
        x = torch.flatten(x, start_dim=1)
        h, c = self.lstm(x, (h, c))
        reward = self.reward_head(h)
        end = self.end_head(h)
        return reward, end, h, c

    def burn_in(self, x: torch.Tensor, actions: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor]:
        batch_size = x.shape[0]
        x = x.flatten(0, 1)
        actions = actions.flatten(0, 1)
        actions_embs = self.action_embedding(actions)
        for res_block in self.trunk:
            x = res_block(x, actions_embs)
        x = x.reshape(batch_size, self.burn_in_len, -1)
        for i in range(self.burn_in_len):
            h, c = self.lstm(x[:, i, :], (h, c))
        return h, c

    def _compute_sin_embeds(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2

        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(
                half_dim,
                device=x.device
            )
            / (half_dim - 1)
        )

        angles = x[:, None] * frequencies[None, :]

        embedding = torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles)
            ],
            dim=-1
        )
        return embedding
    







class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.group_norm = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.activation(self.group_norm(self.conv(x))) + self.skip(x))

class ActorCriticNetwork(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.burn_in_len = args.burn_in_len

        self.trunk = nn.Sequential(
            ResidualBlock(3, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
        )
        self.lstm = nn.LSTMCell(input_size=4*4*64, hidden_size=512)
        self.action_head = nn.Linear(in_features=512, out_features=args.action_space_dim)
        self.value_head = nn.Linear(in_features=512, out_features=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor, compute_value: bool=False) -> Tuple[torch.Tensor | None]:
        x = self.trunk(x)
        x = torch.flatten(x, start_dim=1)
        h, c = self.lstm(x, (h, c))
        action_logits = self.action_head(h)
        if compute_value:
            value = self.value_head(h)
        else:
            value = None
        return action_logits, value,  h, c

    def burn_in(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor]:
        batch_size = x.shape[0]
        x = x.flatten(0, 1)
        x = self.trunk(x)
        x = x.reshape(batch_size, self.burn_in_len, -1)
        for i in range(self.burn_in_len):
            h, c = self.lstm(x[:, i, :], (h, c))
        return h, c


