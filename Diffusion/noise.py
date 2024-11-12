import torch
import torch.nn as nn

class SimpleDiffusion(nn.Module):
    def __init__(
        self,
        num_diffusion_timesteps=1000,
        img_shape=(3, 64, 64),
    ):
        super().__init__()
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.img_shape = img_shape
                
        # BETAs & ALPHAs required at different places in the Algorithm.
        beta = self.get_betas()
        
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', 1 - beta)
        self.register_buffer('sqrt_beta', torch.sqrt(beta))
        self.register_buffer('alpha_cumulative', torch.cumprod(self.alpha, dim=0))
        self.register_buffer('sqrt_alpha_cumulative', torch.sqrt(self.alpha_cumulative))
        self.register_buffer('one_by_sqrt_alpha', 1. / torch.sqrt(self.alpha))
        self.register_buffer('sqrt_one_minus_alpha_cumulative', torch.sqrt(1 - self.alpha_cumulative))

        
    def get_betas(self):
        """linear schedule, proposed in original ddpm paper"""
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start,
            beta_end,
            self.num_diffusion_timesteps,
            dtype=torch.float32,
        )
        
    def forward(self, x0: torch.Tensor, timesteps: torch.Tensor):
        eps     = torch.randn_like(x0)  # Noise
        mean    = get(self.sqrt_alpha_cumulative, t=timesteps) * x0  # Image scaled
        std_dev = get(self.sqrt_one_minus_alpha_cumulative, t=timesteps) # Noise scaled
        sample  = mean + std_dev * eps # scaled inputs * scaled noise

        return sample, eps  # return ... , gt noise --> model predicts this)
    
    def reverse(self, x, ts, z, predicted_noise):
        beta_t                            = get(self.beta, ts)
        one_by_sqrt_alpha_t               = get(self.one_by_sqrt_alpha, ts)
        sqrt_one_minus_alpha_cumulative_t = get(self.sqrt_one_minus_alpha_cumulative, ts) 

        x = (
            one_by_sqrt_alpha_t
            * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
            + torch.sqrt(beta_t) * z
        )
        
