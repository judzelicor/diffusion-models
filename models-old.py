import torch
import math

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import List
from typing import Tuple



class VarianceScheduler(nn.Module):
    def __init__(self, beta_start=0.0001, beta_end=0.02, num_steps=1000, interpolation='linear'):
      super(VarianceScheduler, self).__init__()
      self.num_steps = num_steps
      self.beta_start = beta_start
      self.beta_end = beta_end

      if interpolation == 'linear':
          betas = torch.linspace(beta_start, beta_end, num_steps)
      elif interpolation == 'quadratic':
          betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_steps) ** 2
      else:
          raise NotImplementedError(f'Unknown interpolation {interpolation}')

      alphas = 1.0 - betas
      alpha_cumprod = torch.cumprod(alphas, dim=0)
      alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), alpha_cumprod[:-1]])

      # Precompute values
      sqrt_alphas_cumprod = torch.sqrt(alpha_cumprod)
      sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alpha_cumprod)
      sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
      betas_over_sqrt_one_minus_alphas_cumprod = betas / sqrt_one_minus_alphas_cumprod

      # Register buffers
      self.register_buffer('betas', betas)
      self.register_buffer('alphas', alphas)
      self.register_buffer('alpha_cumprod', alpha_cumprod)
      self.register_buffer('alpha_cumprod_prev', alpha_cumprod_prev)
      self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
      self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
      self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
      self.register_buffer('betas_over_sqrt_one_minus_alphas_cumprod', betas_over_sqrt_one_minus_alphas_cumprod)

    def add_noise(self, x: torch.Tensor, time_step: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
      batch_size = x.size(0)
      device = x.device  # Get the device of x

      # Ensure that self.sqrt_alpha_cumprod is on the same device as x
      sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod.to(device)[time_step].view(batch_size, 1, 1, 1)
      sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(device)[time_step].view(batch_size, 1, 1, 1)

      noise = torch.randn_like(x)

      noisy_x = sqrt_alpha_cumprod_t * x + sqrt_one_minus_alpha_cumprod_t * noise
      return noisy_x, noise


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        """
        Initializes the sinusoidal positional embeddings layer.
        Args:
            dim (int): Dimensionality of the embeddings.
        """
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Computes sinusoidal positional encodings for the input time steps.
        
        Args:
            time (torch.Tensor): Input time steps (batch_size,).
        
        Returns:
            torch.Tensor: Sinusoidal positional embeddings (batch_size, dim).
        """
        device = time.device
        half_dim = self.dim // 2

        # Compute the scaling factor for frequencies
        freq_scale = torch.exp(-torch.arange(half_dim, device=device) * math.log(10000) / (half_dim - 1))
        
        # Scale the time steps
        scaled_time = time[:, None] * freq_scale  # Shape: (batch_size, half_dim)

        # Compute sin and cos embeddings
        sin_emb = torch.sin(scaled_time)
        cos_emb = torch.cos(scaled_time)

        # Concatenate sin and cos embeddings
        embeddings = torch.cat([sin_emb, cos_emb], dim=-1)  # Shape: (batch_size, dim)

        # Handle odd dimensionality by padding with a zero if needed
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros(time.size(0), 1, device=device)], dim=-1)

        return embeddings


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1,
                 down_channels: List[int] = [64, 128, 128, 128, 128],
                 up_channels: List[int] = [128, 128, 128, 128, 64],
                 time_emb_dim: int = 128,
                 num_classes: int = 10) -> None:
        """
        Initializes the UNet model with downsampling, bottleneck, and upsampling layers.
        
        Args:
            in_channels (int): Number of input channels (default=1 for grayscale images).
            down_channels (List[int]): List of channel sizes for downsampling layers.
            up_channels (List[int]): List of channel sizes for upsampling layers.
            time_emb_dim (int): Dimensionality of time embeddings.
            num_classes (int): Number of classes for conditional generation.
        """
        super().__init__()

        self.num_classes = num_classes

        # Time embedding layer
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Label embedding layer
        self.class_emb = nn.Embedding(num_classes, time_emb_dim)

        # Downsampling layers
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        prev_channels = in_channels

        for ch in down_channels:
            self.downs.append(ConvBlock(prev_channels, ch, time_emb_dim))
            prev_channels = ch

        # Bottleneck
        self.bottleneck = ConvBlock(prev_channels, prev_channels, time_emb_dim)

        # Upsampling layers
        self.ups = nn.ModuleList()
        for idx, ch in enumerate(up_channels):
            skip_channels = down_channels[-(idx + 1)]
            self.ups.append(
                nn.ModuleDict({
                    'upconv': nn.ConvTranspose2d(prev_channels, ch, kernel_size=2, stride=2),
                    'convblock': ConvBlock(ch + skip_channels, ch, time_emb_dim)
                })
            )
            prev_channels = ch

        # Final output layer
        self.output_conv = nn.Conv2d(prev_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNet model.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, channels, height, width).
            timestep (torch.Tensor): Timestep tensor for conditioning (batch_size,).
            label (torch.Tensor): Class label tensor for conditional generation (batch_size,).
        
        Returns:
            torch.Tensor: Output tensor (batch_size, channels, height, width).
        """
        # Time and label embeddings
        t_emb = self.time_mlp(timestep)
        l_emb = self.class_emb(label)
        emb = t_emb + l_emb  # Combined embedding for time and label conditioning

        # Downsampling path
        skip_connections = []
        for down in self.downs:
            x = down(x, emb)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x, emb)

        # Upsampling path
        for idx, up in enumerate(self.ups):
            x = up['upconv'](x)
            skip_x = skip_connections[-(idx + 1)]
            x = torch.cat([x, skip_x], dim=1)  # Concatenate skip connections
            x = up['convblock'](x, emb)

        # Output layer
        out = self.output_conv(x)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int) -> None:
        """
        Initializes a convolutional block with time-embedding conditioning.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            emb_dim (int): Dimensionality of the embedding.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_channels)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ConvBlock.
        
        Args:
            x (torch.Tensor): Input feature map.
            emb (torch.Tensor): Time/label embedding.
        
        Returns:
            torch.Tensor: Processed feature map.
        """
        # First convolution
        h = self.conv1(x)
        h = self.act(h)

        # Add time embedding (reshaped for broadcasting)
        emb = self.emb_proj(emb).view(emb.size(0), -1, 1, 1)
        h = h + emb

        # Second convolution
        h = self.conv2(h)
        h = self.act(h)

        return h



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
    def generate_sample(self, num_samples: int, device=torch.device('mps'), labels: torch.Tensor=None):
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
        """
        Initializes the DDPM model.
        
        Args:
            network (nn.Module): The UNet network for noise estimation.
            var_scheduler (VarianceScheduler): The variance scheduler for diffusion steps.
        """
        super().__init__()
        self.var_scheduler = var_scheduler
        self.network = network

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for DDPM: Adds noise to the input and computes the noise prediction loss.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, channels, height, width).
            label (torch.Tensor): Class labels (batch_size,).
        
        Returns:
            torch.Tensor: Loss value.
        """
        # Uniformly sample timesteps for the batch
        batch_size = x.size(0)
        t = torch.randint(0, self.var_scheduler.num_steps, (batch_size,), device=x.device).long()

        # Generate noisy input and the noise added
        noisy_input, noise = self.var_scheduler.add_noise(x, t)

        # Estimate the noise using the network
        estimated_noise = self.network(noisy_input, t, label)

        # Compute loss (L1 loss for DDPM, but L2 loss is common as well)
        loss = F.mse_loss(estimated_noise, noise)

        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
      """
      Recover the sample at the previous timestep based on DDPM sampling strategy.
      """
      device = timestep.device  # Ensure consistency with timestep device

      # Move beta_t and alpha_bar_t to the same device as timestep
      beta_t = self.var_scheduler.betas.to(device)[timestep].view(-1, 1, 1, 1)
      alpha_bar_t = self.var_scheduler.alpha_cumprod.to(device)[timestep].view(-1, 1, 1, 1)

      # Compute the mean of the posterior distribution
      mean = (noisy_sample - torch.sqrt(1 - alpha_bar_t) * estimated_noise) / torch.sqrt(alpha_bar_t)

      # Add noise for intermediate timesteps
      if timestep.min() > 0:
          noise = torch.randn_like(noisy_sample, device=device)
          sample = mean + torch.sqrt(beta_t) * noise
      else:
          sample = mean  # No noise added for t=0

      return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device = torch.device('cuda'), labels: torch.Tensor = None) -> torch.Tensor:
        """
        Iteratively generates samples using the DDPM reverse process.
        
        Args:
            num_samples (int): Number of samples to generate.
            device (torch.device): Device for computation (default: 'cuda').
            labels (torch.Tensor): Optional class labels (batch_size,).
        
        Returns:
            torch.Tensor: Generated samples (num_samples, channels, height, width).
        """
        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, "Error: number of labels must match number of samples!"
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, (num_samples,), device=device)
        else:
            labels = None

        # Start from pure noise
        sample = torch.randn((num_samples, 1, 32, 32), device=device)

        # Reverse the diffusion process
        for t in reversed(range(self.var_scheduler.num_steps)):
            timestep = torch.tensor([t] * num_samples, device=device).long()
            estimated_noise = self.network(sample, timestep, labels)
            sample = self.recover_sample(sample, estimated_noise, timestep)

        return sample


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
    
