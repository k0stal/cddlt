# Shi et al. (2016)

import torch
import dl_tools

class ESPCN(dl_tools.DLModule):
    def __init__(self, n_channels: int, upscale_factor: int) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(n_channels, 64, kernel_size=5, padding=2),
            torch.nn.Tanh(),
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.Tanh(),
            torch.nn.Conv2d(32, n_channels * (upscale_factor**2), kernel_size=3, padding=1),
            torch.nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)