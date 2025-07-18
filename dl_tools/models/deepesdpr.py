import torch
import dl_tools

class DeepESDpr(dl_tools.DLModule):

    """
    DeepESD model as proposed in Baño-Medina et al. 2024 for precipitation
    downscaling. This implementation allows for a deterministic (MSE-based)
    and stochastic (NLL-based) definition.

    Baño-Medina, J., Manzanas, R., Cimadevilla, E., Fernández, J., González-Abad,
    J., Cofiño, A. S., and Gutiérrez, J. M.: Downscaling multi-model climate projection
    ensembles with deep learning (DeepESD): contribution to CORDEX EUR-44, Geosci. Model
    Dev., 15, 6747–6758, https://doi.org/10.5194/gmd-15-6747-2022, 2022.

    Parameters
    ----------
    x_shape : tuple
        Shape of the data used as predictor. This must have dimension 4
        (time, channels/variables, lon, lat).

    y_shape : tuple
        Shape of the data used as predictand. This must have dimension 2
        (time, gridpoint)

    filters_last_conv : int
        Number of filters/kernels of the last convolutional layer

    stochastic: bool
        If set to True, the model is composed of three final dense layers computing
        the p, shape and scale of the Bernoulli-gamma distribution. Otherwise,
        the models is composed of one final layer computing the values.

    last_relu: bool, optional
        If set to True, the output of the last dense layer is passed through a
        ReLU activation function. This does not apply when stochastic=True. By
        default is set to False.
    """

    def __init__(self, x_shape: tuple, y_shape: tuple,
                 filters_last_conv: int, stochastic: bool,
                 last_relu: bool=False):

        super(DeepESDpr, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            error_msg =\
            'X and Y data must have a dimension of length 4'
            'and 2, correspondingly'

            raise ValueError(error_msg)

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv
        self.stochastic = stochastic
        self.last_relu = last_relu

        self.conv_1 = torch.nn.Conv2d(in_channels=self.x_shape[1],
                                      out_channels=50,
                                      kernel_size=3,
                                      padding=1)

        self.conv_2 = torch.nn.Conv2d(in_channels=50,
                                      out_channels=25,
                                      kernel_size=3,
                                      padding=1)

        self.conv_3 = torch.nn.Conv2d(in_channels=25,
                                      out_channels=self.filters_last_conv,
                                      kernel_size=3,
                                      padding=1)

        if self.stochastic:
            self.p = torch.nn.Linear(in_features=\
                                            self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                            out_features=self.y_shape[1])

            self.log_shape = torch.nn.Linear(in_features=\
                                             self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                             out_features=self.y_shape[1])

            self.log_scale = torch.nn.Linear(in_features=\
                                             self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                             out_features=self.y_shape[1])

        else:
            self.out = torch.nn.Linear(in_features=\
                                       self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                       out_features=self.y_shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_1(x)
        x = torch.relu(x)

        x = self.conv_2(x)
        x = torch.relu(x)

        x = self.conv_3(x)
        x = torch.relu(x)

        x = torch.flatten(x, start_dim=1)

        if self.stochastic:
            p = self.p(x)
            p = torch.sigmoid(p)

            log_shape = self.log_shape(x)
            log_scale = self.log_scale(x)

            out = torch.cat((p, log_shape, log_scale), dim = 1)
        else:
            out = self.out(x)
            if self.last_relu: out = torch.relu(out)

        return out