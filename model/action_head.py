"""
Implementations of MLP and diffusion action heads, which serve as alternatives to VLM sequential token prediction.
Adapted from
https://github.com/moojink/openvla-oft/blob/main/prismatic/models/action_heads.py
https://github.com/microsoft/CogACT/blob/main/action_model/action_model.py
"""

import math

import torch
import torch.nn as nn

from .diffusion.models import DiT
from .diffusion import create_diffusion
from .diffusion import gaussian_diffusion as gd


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""

    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.gelu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""

    def __init__(
        self,
        num_blocks=1,
        input_dim=2304,
        hidden_dim=512,
        action_dim=5,
        output_dim=1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(num_blocks=num_blocks, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, self.action_dim, -1)
        action = self.model(rearranged_actions_hidden_states)
        action = action.squeeze()
        return action


def DiT_SS(**kwargs):
    return DiT(depth=3, hidden_size=256, num_heads=4, **kwargs)
def DiT_S(**kwargs):
    return DiT(depth=6, hidden_size=384, num_heads=4, **kwargs)
def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)
def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)


DiT_models = {"DiT-SS": DiT_SS, "DiT-S": DiT_S, "DiT-B": DiT_B, "DiT-L": DiT_L}


class DiffusionActionHead(nn.Module):
    def __init__(
        self,
        token_size,
        model_type,
        in_channels,
        future_action_window_size,
        past_action_window_size,
        diffusion_steps=100,
        noise_schedule="squaredcos_cap_v2",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.noise_schedule = noise_schedule
        # GaussianDiffusion offers forward and backward functions q_sample and p_sample.
        self.diffusion_steps = diffusion_steps
        self.diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule=noise_schedule,
            diffusion_steps=self.diffusion_steps,
            sigma_small=True,
            learn_sigma=False,
        )
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [
            gd.ModelVarType.LEARNED,
            gd.ModelVarType.LEARNED_RANGE,
        ]:
            learn_sigma = True
        else:
            learn_sigma = False
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.net = DiT_models[model_type](
            token_size=token_size,
            in_channels=in_channels,
            class_dropout_prob=0.1,
            learn_sigma=learn_sigma,
            future_action_window_size=future_action_window_size,
            past_action_window_size=past_action_window_size,
        )

    # Given condition z and ground truth token x, compute loss
    def loss(self, x, z):
        # sample random noise and timestep
        noise = torch.randn_like(x)  # [B, T, C]
        timestep = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device=x.device)

        # sample x_t from x
        x_t = self.diffusion.q_sample(x, timestep, noise)

        # predict noise from x_t
        noise_pred = self.net(x_t, timestep, z)

        assert noise_pred.shape == noise.shape == x.shape
        # Compute L2 loss
        loss = ((noise_pred - noise) ** 2).mean()
        # Optional: loss += loss_vlb

        return loss

    def get_action(self, action_hidden_states):
        self.eval()

        action_hidden_states = action_hidden_states.permute(0, 2, 1).mean(dim=1, keepdim=True)  # [B, 5, 2304] -> [B, 1, 5]

        # Sample random noise
        noise = torch.randn_like(action_hidden_states)

        model_kwargs = dict(z=action_hidden_states)
        sample_fn = self.net.forward

        if self.ddim_diffusion is None:
            self.create_ddim(ddim_step=10)

        samples = self.ddim_diffusion.ddim_sample_loop(
            sample_fn,
            noise.shape,
            noise,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=action_hidden_states.device,
            eta=0.0,
        )
        actions = samples[:, -1]  # [B, T, action_dim]
        # actions = torch.clamp(actions, -1.0, 1.0)  # optional

        return actions.detach()

    # Create DDIM sampler
    def create_ddim(self, ddim_step=10):
        self.ddim_diffusion = create_diffusion(
            timestep_respacing="ddim" + str(ddim_step),
            noise_schedule=self.noise_schedule,
            diffusion_steps=self.diffusion_steps,
            sigma_small=True,
            learn_sigma=False,
        )
        return self.ddim_diffusion
