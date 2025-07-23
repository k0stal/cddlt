import torch
import cddlt

class Bicubic(cddlt.DLModule):
    def __init__(self, upscale_factor: int) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(
            x,
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False,
            antialias=True
        )

        return x