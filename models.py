import torch
import math

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import List
from typing import Tuple



class VarianceScheduler:
    def __init__(self, beta_start: int=0.0001, beta_end: int=0.02, num_steps: int=1000, interpolation: str='linear') -> None:
        self.num_steps = num_steps

        # find the beta valuess by linearly interpolating from start beta to end beta
        if interpolation == 'linear':
            # TODO: complete the linear interpolation of betas here
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif interpolation == 'quadratic':
            # TODO: complete the quadratic interpolation of betas here
            sqrt_beta_start = math.sqrt(beta_start)
            sqrt_beta_end = math.sqrt(beta_end)
            sqrt_betas = torch.linspace(sqrt_beta_start, sqrt_beta_end, num_steps)
            self.betas = sqrt_betas ** 2
        else:
            raise Exception('[!] Error: invalid beta interpolation encountered...')
        

        # TODO: add other statistics such alphas alpha_bars and all the other things you might need here
        # Calculate alphas and cumulative products (alpha_bars)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x:torch.Tensor, time_step:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device

        # Sample random noise
        noise = torch.randn_like(x)

        # Ensure self.alpha_bars is on the same device as time_step
        alpha_bars = self.alpha_bars.to(device)

        # Retrieve alpha_bar_t for the given time steps
        alpha_bar_t = alpha_bars[time_step].view(-1, 1, 1, 1)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

        # Generate the noisy input
        noisy_x = sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise

        return noisy_x, noise


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
      super().__init__()

      self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # TODO: compute the sinusoidal positional encoding of the time
        device = time.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(-torch.arange(half_dim, device=device) * emb_scale)
        emb = time[:, None] * emb[None, :]  # Shape: [batch_size, half_dim]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # Shape: [batch_size, dim]
        return emb
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.act2 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Time and label embeddings projection
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.label_emb_proj = nn.Linear(time_emb_dim, out_channels)

        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, l_emb: torch.Tensor) -> torch.Tensor:
        # First layer
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # Add time and label embeddings
        t_proj = self.time_emb_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        l_proj = self.label_emb_proj(l_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t_proj + l_proj

        # Second layer
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        # Residual connection
        return h + self.shortcut(x)


# class UNet(nn.Module):
#     def __init__(self, in_channels: int=1, 
#                  down_channels: List=[64, 128, 128, 128, 128], 
#                  up_channels: List=[128, 128, 128, 128, 64], 
#                  time_emb_dim: int=128,
#                  num_classes: int=10) -> None:
#         super().__init__()

#         # # NOTE: You can change the arguments received by the UNet if you want, but keep the num_classes argument

#         # self.num_classes = num_classes

#         # # TODO: time embedding layer
#         # self.time_mlp = ...

#         # # TODO: define the embedding layer to compute embeddings for the labels
#         # self.class_emb = ...

#         # # define your network architecture here
#         # Time embedding layer
#         self.num_classes = num_classes

#         # Time embedding layer
#         self.time_mlp = nn.Sequential(
#             SinusoidalPositionEmbeddings(time_emb_dim),
#             nn.Linear(time_emb_dim, time_emb_dim * 4),
#             nn.SiLU(),
#             nn.Linear(time_emb_dim * 4, time_emb_dim)
#         )

#         # Label embedding layer
#         self.class_emb = nn.Embedding(num_classes, time_emb_dim)

#         # Initial convolution
#         self.init_conv = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, padding=1)

#         # Downsampling layers
#         self.downs = nn.ModuleList()
#         num_downs = len(down_channels)
#         for i in range(num_downs - 1):
#             self.downs.append(nn.ModuleDict({
#                 'block1': ResidualBlock(down_channels[i], down_channels[i+1], time_emb_dim),
#                 'block2': ResidualBlock(down_channels[i+1], down_channels[i+1], time_emb_dim),
#                 'downsample': nn.Conv2d(down_channels[i+1], down_channels[i+1], kernel_size=3, stride=2, padding=1)
#             }))

#         # Bottleneck layers
#         self.bottleneck = nn.ModuleList([
#             ResidualBlock(down_channels[-1], down_channels[-1], time_emb_dim),
#             ResidualBlock(down_channels[-1], down_channels[-1], time_emb_dim)
#         ])

#         # Upsampling layers
#         self.ups = nn.ModuleList()
#         num_ups = len(up_channels)
#         for i in range(num_ups - 1):
#             # Calculate in_channels for block1 after concatenation
#             in_channels_block1 = up_channels[i+1] + down_channels[-(i+2)]
#             self.ups.append(nn.ModuleDict({
#                 'upsample': nn.ConvTranspose2d(up_channels[i], up_channels[i+1], kernel_size=2, stride=2),
#                 'block1': ResidualBlock(
#                     in_channels=in_channels_block1,
#                     out_channels=up_channels[i+1],
#                     time_emb_dim=time_emb_dim
#                 ),
#                 'block2': ResidualBlock(
#                     in_channels=up_channels[i+1],
#                     out_channels=up_channels[i+1],
#                     time_emb_dim=time_emb_dim
#                 )
#             }))

#         # Final convolution
#         self.final_conv = nn.Sequential(
#             nn.GroupNorm(num_groups=32, num_channels=up_channels[-1]),
#             nn.SiLU(),
#             nn.Conv2d(up_channels[-1], in_channels, kernel_size=3, padding=1)
#         )

#     def forward(self, x: torch.Tensor, timestep: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
#         # # TODO: embed time
#         # t = ...

#         # # TODO: handle label embeddings if labels are avaialble
#         # l = ...
        
#         # # TODO: compute the output of your network
#         # out = ...

#         # return out
#         batch_size = x.size(0)
#         device = x.device

#         # Embed time
#         t_emb = self.time_mlp(timestep.to(device).float())  # Shape: [batch_size, time_emb_dim]

#         # Label embedding
#         l_emb = self.class_emb(label.to(device))  # Shape: [batch_size, time_emb_dim]

#         # Initial convolution
#         x = self.init_conv(x)

#         # Store skip connections
#         skip_connections = []

#         # Downsampling path
#         for down in self.downs:
#             x = down['block1'](x, t_emb, l_emb)
#             x = down['block2'](x, t_emb, l_emb)
#             skip_connections.append(x)
#             x = down['downsample'](x)

#         # Bottleneck
#         for block in self.bottleneck:
#             x = block(x, t_emb, l_emb)

#         # Reverse the skip connections for upsampling
#         skip_connections = skip_connections[::-1]

#         # Upsampling path
#         for i, up in enumerate(self.ups):
#             x = up['upsample'](x)
#             skip = skip_connections[i]

#             # Concatenate skip connection
#             x = torch.cat([x, skip], dim=1)

#             # Apply blocks
#             x = up['block1'](x, t_emb, l_emb)
#             x = up['block2'](x, t_emb, l_emb)

#         # Final convolution
#         out = self.final_conv(x)

#         return out  # Estimated noise level

class UNet(nn.Module):
    def __init__(self, in_channels: int=1, 
                 down_channels: List[int]=[64, 128, 128, 128, 128], 
                 up_channels: List[int]=[128, 128, 128, 128, 64], 
                 time_emb_dim: int=128,
                 num_classes: int=10) -> None:
        super().__init__()

        self.num_classes = num_classes

        # Time embedding layer
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Label embedding layer
        self.class_emb = nn.Embedding(num_classes, time_emb_dim)

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, padding=1)

        # Downsampling layers
        self.downs = nn.ModuleList()
        num_downs = len(down_channels)
        for i in range(num_downs - 1):
            self.downs.append(nn.ModuleDict({
                'block1': ResidualBlock(down_channels[i], down_channels[i+1], time_emb_dim),
                'block2': ResidualBlock(down_channels[i+1], down_channels[i+1], time_emb_dim),
                'downsample': nn.Conv2d(down_channels[i+1], down_channels[i+1], kernel_size=3, stride=2, padding=1)
            }))

        # Bottleneck layers
        self.bottleneck = nn.ModuleList([
            ResidualBlock(down_channels[-1], down_channels[-1], time_emb_dim),
            ResidualBlock(down_channels[-1], down_channels[-1], time_emb_dim)
        ])

        # Compute skip_channels (channels from skip connections)
        skip_channels = down_channels[1:][::-1]  # Reverse the list to match upsampling order

        # Upsampling layers
        self.ups = nn.ModuleList()
        num_ups = len(up_channels)
        for i in range(num_ups - 1):
            in_channels_block1 = up_channels[i+1] + skip_channels[i]  # Adjusted in_channels
            self.ups.append(nn.ModuleDict({
                'upsample': nn.ConvTranspose2d(up_channels[i], up_channels[i+1], kernel_size=2, stride=2),
                'block1': ResidualBlock(
                    in_channels=in_channels_block1,
                    out_channels=up_channels[i+1],
                    time_emb_dim=time_emb_dim
                ),
                'block2': ResidualBlock(
                    in_channels=up_channels[i+1],
                    out_channels=up_channels[i+1],
                    time_emb_dim=time_emb_dim
                )
            }))

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=up_channels[-1]),
            nn.SiLU(),
            nn.Conv2d(up_channels[-1], in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device

        # Embed time
        t_emb = self.time_mlp(timestep.to(device).float())  # Shape: [batch_size, time_emb_dim]

        # Label embedding
        l_emb = self.class_emb(label.to(device))  # Shape: [batch_size, time_emb_dim]

        # Initial convolution
        x = self.init_conv(x)

        # Store skip connections
        skip_connections = []

        # Downsampling path
        for down in self.downs:
            x = down['block1'](x, t_emb, l_emb)
            x = down['block2'](x, t_emb, l_emb)
            skip_connections.append(x)
            x = down['downsample'](x)

        # Bottleneck
        for block in self.bottleneck:
            x = block(x, t_emb, l_emb)

        # Reverse the skip connections for upsampling
        skip_connections = skip_connections[::-1]

        # Upsampling path
        for i, up in enumerate(self.ups):
            x = up['upsample'](x)
            skip = skip_connections[i]

            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)

            # Apply blocks
            x = up['block1'](x, t_emb, l_emb)
            x = up['block2'](x, t_emb, l_emb)

        # Final convolution
        out = self.final_conv(x)

        return out  # Estimated noise level


class VAE(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 height: int=32, 
                 width: int=32, 
                 mid_channels: List=[32, 32, 32], 
                 latent_dim: int=32, 
                 num_classes: int=10) -> None:
        
        super().__init__()

        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # NOTE: self.mid_size specifies the size of the image [C, H, W] in the bottleneck of the network
        self.mid_size = [mid_channels[-1], height // (2 ** (len(mid_channels)-1)), width // (2 ** (len(mid_channels)-1))]

        # NOTE: You can change the arguments of the VAE as you please, but always define self.latent_dim, self.num_classes, self.mid_size
        
        # TODO: handle the label embedding here
        self.class_emb = ...
        
        # TODO: define the encoder part of your network
        self.encoder = ...
        
        # TODO: define the network/layer for estimating the mean
        self.mean_net = ...
        
        # TODO: define the networklayer for estimating the log variance
        self.logvar_net = ...

        # TODO: define the decoder part of your network
        self.decoder = ...
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # TODO: compute the output of the network encoder
        out = ...

        # TODO: estimating mean and logvar
        mean = self.mean_net(out)
        logvar = self.logvar_net(out)
        
        # TODO: computing a sample from the latent distribution
        sample = self.reparameterize(mean, logvar)

        # TODO: decoding the sample
        out = self.decode(sample, label)

        return out, mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: implement the reparameterization trick: sample = noise * std + mean
        sample = ...

        return sample
    
    @staticmethod
    def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # TODO: compute the binary cross entropy between the pred (reconstructed image) and the traget (ground truth image)
        loss = F.binary_cross_entropy(pred, target, reduction='sum')

        return loss
       
    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: compute the KL divergence
        kl_div = -.5 * (logvar.flatten(start_dim=1) + 1 - torch.exp(logvar.flatten(start_dim=1)) - mean.flatten(start_dim=1).pow(2)).sum()

        return kl_div

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        else:
            # randomly consider some labels
            labels = torch.randint(0, self.num_classes, [num_samples,], device=device)

        # TODO: sample from standard Normal distrubution
        noise = ...

        # TODO: decode the noise based on the given labels
        out = ...

        return out
    
    def decode(self, sample: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # TODO: use you decoder to decode a given sample and their corresponding labels
        out = ...

        return out


class LDDPM(nn.Module):
    def __init__(self, network: nn.Module, vae: VAE, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.vae = vae
        self.network = network

        # freeze vae
        self.vae.requires_grad_(False)
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # TODO: uniformly sample as many timesteps as the batch size
        t = ...

        # TODO: generate the noisy input
        noisy_input, noise = ...

        # TODO: estimate the noise
        estimated_noise = ...

        # compute the loss (either L1 or L2 loss)
        loss = F.mse_loss(estimated_noise, noise)

        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # TODO: implement the sample recovery strategy of the DDPM
        sample = ...

        return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        else:
            labels = torch.randint(0, self.vae.num_classes, [num_samples,], device=device)
        
        # TODO: using the diffusion model generate a sample inside the latent space of the vae
        # NOTE: you need to recover the dimensions of the image in the latent space of your VAE
        sample = ...

        sample = self.vae.decode(sample, labels)
        
        return sample


class DDPM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # # TODO: uniformly sample as many timesteps as the batch size
        # t = ...

        # # TODO: generate the noisy input
        # noisy_input, noise = ...

        # # TODO: estimate the noise
        # estimated_noise = ...

        # # TODO: compute the loss (either L1, or L2 loss)
        # loss = F.l1_loss(estimated_noise, noise)

        # return loss
        device = x.device
        batch_size = x.size(0)

        # Uniformly sample as many timesteps as the batch size
        t = torch.randint(0, self.var_scheduler.num_steps, (batch_size,), device=device).long()

        # Generate the noisy input
        noisy_input, noise = self.var_scheduler.add_noise(x, t)

        # Estimate the noise
        estimated_noise = self.network(noisy_input, t, label)

        # Compute the loss (either L1, or L2 loss)
        loss = F.l1_loss(estimated_noise, noise)

        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # # TODO: implement the sample recovery strategy of the DDPM
        # sample = ...

        # return sample
        device = noisy_sample.device
        betas = self.var_scheduler.betas.to(device)
        alphas = self.var_scheduler.alphas.to(device)
        alpha_bars = self.var_scheduler.alpha_bars.to(device)

        # Get the required parameters for the given timestep
        beta_t = betas[timestep].view(-1, 1, 1, 1)
        alpha_t = alphas[timestep].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bars[timestep]).view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = 1 / torch.sqrt(alpha_t)

        # Recover the previous sample
        sample = sqrt_recip_alpha_t * (noisy_sample - beta_t / sqrt_one_minus_alpha_bar_t * estimated_noise)

        return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):
        # if labels is not None and self.network.num_classes is not None:
        #     assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
        #     labels = labels.to(device)
        # elif labels is None and self.network.num_classes is not None:
        #     labels = torch.randint(0, self.network.num_classes, [num_samples,], device=device)
        # else:
        #     labels = None

        # # TODO: apply the iterative sample generation of the DDPM
        # sample = ...

        # return sample
        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, (num_samples,), device=device)
        else:
            labels = None

        # Initialize the sample with random noise
        x = torch.randn(num_samples, self.network.init_conv.in_channels, 32, 32, device=device)
        num_steps = self.var_scheduler.num_steps

        for t in reversed(range(num_steps)):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

            # Estimate the noise at timestep t
            estimated_noise = self.network(x, t_batch, labels)

            # Get parameters for timestep t
            beta_t = self.var_scheduler.betas[t].to(device)
            alpha_t = self.var_scheduler.alphas[t].to(device)
            alpha_bar_t = self.var_scheduler.alpha_bars[t].to(device)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
            sqrt_recip_alpha_t = 1 / torch.sqrt(alpha_t)

            beta_t = beta_t.view(1, 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.view(1, 1, 1, 1)
            sqrt_recip_alpha_t = sqrt_recip_alpha_t.view(1, 1, 1, 1)

            # Compute the mean of the posterior q(x_{t-1} | x_t, x_0)
            x = sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_alpha_bar_t * estimated_noise)

            # Add noise if not the last timestep
            if t > 0:
                noise = torch.randn_like(x)
                beta_t_prev = self.var_scheduler.betas[t - 1].to(device).view(1, 1, 1, 1)
                sigma_t = torch.sqrt(beta_t_prev)
                x = x + sigma_t * noise

        return x  # Generated samples


class DDIM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: uniformly sample as many timesteps as the batch size
        t = ...

        # TODO: generate the noisy input
        noisy_input, noise = ...

        # TODO: estimate the noise
        estimated_noise = ...

        # TODO: compute the loss
        loss = F.l1_loss(estimated_noise, noise)

        return loss
    
    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # TODO: apply the sample recovery strategy of the DDIM
        sample = ...

        return sample
    
    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, [num_samples,], device=device)
        else:
            labels = None
        # TODO: apply the iterative sample generation of DDIM (similar to DDPM)
        sample = ...

        return sample
    
