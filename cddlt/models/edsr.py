import sympy
import torch.nn
import cddlt

class ResidualBlock(torch.nn.Module):
    def __init__(self, features):
        super().__init__()
        self.residual_block = torch.nn.Sequential(
            torch.nn.Conv2d(features, features, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(features, features, kernel_size=3, padding=1),
        )

    def forward(self, x):
        residue = self.residual_block(x)
        # residual scaling
        # TODO add reference
        residue = residue.mul(0.1)
        residue += x
        return residue


class Upsampler(torch.nn.Sequential):
    def __init__(self, scale_factor, features):
        # TODO ascending or descending order?
        prime_factors = sympy.factorint(scale_factor, multiple=True)
        modules = []
        for prime_factor in prime_factors:
            modules.append(
                torch.nn.Conv2d(
                    features, features * (prime_factor**2), kernel_size=3, padding=1
                )
            )
            modules.append(torch.nn.PixelShuffle(prime_factor))
        super().__init__(*modules)


class EDSR(cddlt.DLModule):
    # official implementation:
    # https://github.com/sanghyun-son/EDSR-PyTorch/tree/master
    def __init__(self, channels, scale_factor):
        super().__init__()
        # Lim et al. (2017):
        # "We train our networks using L1 loss instead of L2. Minimizing L2 is
        # generally preferred since it maximizes the PSNR. However, based on
        # a series of experiments we empirically found that L1 loss provides
        # better convergence than L2. The evaluation of this comparison is
        # provided in Sec. 4.4"
        self.loss = torch.nn.L1Loss()
        features = 256
        residual_blocks = 32
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(channels, features, kernel_size=3, padding=1)
        )
        self.body = torch.nn.Sequential(
            *[ResidualBlock(features) for _ in range(residual_blocks)],
            torch.nn.Conv2d(features, features, kernel_size=3, padding=1),
        )
        self.tail = torch.nn.Sequential(
            Upsampler(scale_factor, features),
            torch.nn.Conv2d(features, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.head(x)
        residue = self.body(x)
        residue += x
        x = self.tail(residue)
        return x