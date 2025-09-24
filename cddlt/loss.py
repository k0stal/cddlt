import torch

"""
Implementation of losses for determinstic predictions.
All are inherited from general `torch.nn.Module`.
"""

class NNLGammaLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(NNLGammaLoss, self).__init__()
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        input:  [batch_size, shape || scale]
        target: [batch_size, grid_size]
        """

        grid_size = target.shape[0]
        prob = input[:, grid_size:]
        shape = input[:, grid_size:(2 * grid_size)]
        scale = input[:, (2 * grid_size):]

        loss = \
            (shape * torch.log(scale) + torch.lgamma(shape) + target / scale) / \
            (torch.log(prob) + (shape - 1) * torch.log(target))
        
        return -torch.mean(loss)

class NNLBernoulliGammaLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(NNLBernoulliGammaLoss, self).__init__()
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        input:  [batch_size, probability || shape || scale]
        target: [batch_size, grid_size]
        """
        
        grid_size = target.shape[0]
        prob = input[:, grid_size:]
        shape = input[:, grid_size:(2 * grid_size)]
        scale = input[:, (2 * grid_size):]

        zero_mask = [input == 0].type(torch.float32)
        zero_case = zero_mask * torch.log(1 - prob)

        nonzero_case = ~zero_mask * \
            (shape * torch.log(scale) + torch.lgamma(shape) + target / scale) / \
            (torch.log(prob) + (shape - 1) * torch.log(target))
        
        return -torch.mean(zero_case + nonzero_case)

"""
ASYM Calculate distribution beforehand.
"""

class ASYMmetricLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(ASYMmetricLoss, self).__init__()
        raise NotImplementedError()
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
