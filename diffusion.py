from typing import Tuple
import math
import numpy as np
import torch
import torch.nn as nn

class AdaGroupNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Linear(in_features=256, out_features=64)
        self.shift = nn.Linear(in_features=256, out_features=64)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=64) 

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = self.scale(c)
        shift = self.shift(c)
        x = self.norm(x)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return (1 + scale) * x + shift


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
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
        return out + res

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.res_block = ResBlock(in_channels=in_channels, out_channels=out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor]:
        skip = self.res_block(x, c)
        out = self.pool(skip)
        return out, skip

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.upsample=  nn.ConvTranspose2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

        self.res_block = ResBlock(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        cat_x = torch.concatenate([x, skip], dim=1)
        out = self.res_block(cat_x, c)
        return out

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.buffer = []
        self.down_blocks = nn.ModuleList([DownBlock(in_channels=15 if (i==0) else 64, out_channels=64) for i in range(4)])
        self.bottleneck = ResBlock(in_channels=64, out_channels=64)
        self.up_blocks = nn.ModuleList([UpBlock(in_channels=128, out_channels=64) for _ in range(4)])
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

    def forward(self, x: torch.Tensor, tau_embs: torch.Tensor, action_embs: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        x = torch.concatenate([self.buffer, x.unsqueeze(1)], dim=1).flatten(start_dim=1, end_dim=2)
        c = torch.concatenate([tau_embs, action_embs], dim=-1)
        for  i, down_block in enumerate(self.down_blocks):
            x, skip = down_block(x, c)
            skip_connections.append(skip)

        x = self.bottleneck(x, c)

        for up_block in self.up_blocks:
            x = up_block(x, skip_connections.pop(), c)

        return self.output(x)

    def setup_buffer(self, frames):
        self.buffer = frames

class EDMDiffusionModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = 128
        self.sigma_max = 5
        self.sigma_min = 0.002
        self.ro = 7
        self.S_noise = 1.003
        self.S_churn = 0 #40
        self.S_tmin = 0.05
        self.S_tmax = 50
        self.n_sampling_steps = args.n_sampling_steps
        self.batch_size = args.batch_size
        self.time_steps = [(self.sigma_max**(1/self.ro) + i/(self.n_sampling_steps-1)*(self.sigma_min**(1/self.ro) - self.sigma_max**(1/self.ro)))**self.ro for i in range(self.n_sampling_steps)] + [0]
        self.action_embedding = nn.Embedding(args.action_space_dim, self.embed_dim //4)
        self.denoiser = Unet()

    def forward(self, noised_target: torch.Tensor, past_frames: torch.Tensor, sigma_tau: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor]:
        c_in, c_out, c_noise, c_skip = self._compute_precons(sigma_tau)
        c_noise_embs = self._compute_sin_embeds(c_noise)
        action_embs = self.action_embedding(actions).flatten(start_dim=1)
        self.denoiser.setup_buffer(past_frames/0.5)
        denoised_target = self.denoiser(c_in*noised_target, c_noise_embs, action_embs)
        return c_skip * noised_target + c_out * denoised_target, c_out

    def sample(self, past_frames: torch.Tensor, past_actions: torch.Tensor) -> torch.Tensor:
        x = self.time_steps[0] * torch.randn_like(past_frames[:, 0, :, :, :])
        for i in range(self.n_sampling_steps):
            t_step = self.time_steps[i]
            eps_i = self.S_noise * torch.randn_like(x) 
            gamma_i = np.min([self.S_churn/self.n_sampling_steps, np.sqrt(2)-1]) if ((t_step >= self.S_tmin) and (t_step <= self.S_tmax)) else 0
            noised_t_step = (1+gamma_i)*t_step
            x_hat = x + np.sqrt(noised_t_step**2 - t_step**2) * eps_i
            sigma_tau = torch.full((self.batch_size,), noised_t_step, device="cuda")
            denoised_x_hat, _ = self.forward(x_hat, past_frames, sigma_tau, past_actions)
            d_i = (x_hat - denoised_x_hat) / noised_t_step
            x = x_hat + d_i * (self.time_steps[i+1] - noised_t_step)
        return x.clamp(-1, 1)

    def _compute_precons(self, sigma_tau: torch.Tensor) -> Tuple[torch.Tensor]:
        c_in = 1/torch.sqrt(sigma_tau**2 + 0.5**2)
        c_out = 0.5 * sigma_tau / torch.sqrt(sigma_tau**2 + 0.5**2)
        c_noise = 1/4*torch.log(sigma_tau)
        c_skip = 0.5**2 / (sigma_tau**2 + 0.5**2)
        c_in = c_in.view(-1, 1, 1, 1).to("cuda")
        c_out = c_out.view(-1, 1, 1, 1)
        c_noise = c_noise.to("cuda")
        c_skip = c_skip.view(-1, 1, 1, 1)
        return c_in, c_out, c_noise, c_skip

    def _compute_sin_embeds(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2

        frequencies = torch.exp(-math.log(10000) * torch.arange(half_dim, device=x.device) / (half_dim - 1))
        angles = x[:, None] * frequencies[None, :]
        embedding = torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles)
            ],
            dim=-1
        )
        return embedding
