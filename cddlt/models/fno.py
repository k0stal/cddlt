import cddlt
import torch

class SpectralConv2d(cddlt.DLModule):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        # Directly parameterizing R in Fourier space
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = torch.nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = torch.nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]

        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
                            device=x.device, dtype=torch.cfloat)

        out_ft[:, :, :self.modes1, :self.modes2] = self._complex_matmul(
            x_ft[:, :, :self.modes1, :self.modes2],
            torch.view_as_complex(self.weights1)
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self._complex_matmul(
            x_ft[:, :, -self.modes1:, :self.modes2],
            torch.view_as_complex(self.weights2)
        )

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

    def _complex_matmul(self, x, weights):
        return torch.einsum("bixy,ioxy->boxy", x, weights)

class FNO(cddlt.DLModule):
    def __init__(
        self, 
        n_channels: int, 
        upscale_factor: int, 
        layers: int = 4, 
        width: int = 32, 
        modes1: int = 12, 
        modes2: int = 12
    ) -> None:
        
        super(FNO, self).__init__()

        self.upscale_factor = upscale_factor
        self.layers = layers
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.P = torch.nn.Linear(n_channels, self.width)

        self.spectral_convs = torch.nn.ModuleList(
            [SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(layers)]
        )

        self.weights = torch.nn.ModuleList(
            [torch.nn.Conv2d(self.width, self.width, kernel_size=1) for _ in range(layers)]
        )

        self.Q = torch.nn.Linear(self.width, n_channels)

    def forward(self, x: torch.Tensor) -> None:

        x = torch.nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic')

        # [B, C, W, H] to [B, W, H, C]
        x = x.permute(0, 2, 3 ,1)
        x = self.P(x)

        # [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        for (spconv, weight) in zip(self.spectral_convs, self.weights):
            x1 = spconv(x)
            x2 = weight(x)
            x = x1 + x2
            x = torch.nn.functional.relu(x)

        # [B, C, W, H] to [B, W, H, C]
        x = x.permute(0, 2, 3, 1)
        x = self.Q(x)

        # [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()