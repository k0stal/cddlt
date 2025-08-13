import torch
import cddlt

class Bicubic(cddlt.DLModule):
    def __init__(self, upscale_factor: int, mode: str = "bicubic") -> None:
        super().__init__()
        self.upscale_factor = upscale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(
            x,
            scale_factor=self.upscale_factor,
            mode=self.mode,
            align_corners=False,
            antialias=True
        )

        return x