import torch 
from typing import Set, List

"""
    #TODO
    - implement methods and validations for evaluation
    - store to tensorboard
"""


class VALUE:

    """Supported variables."""
    VARIABLES: Set[str] = {"PR", "TM", "TN", "TX"}

    class Variable:
        def __init__(
            self,
            prediction: List[torch.Tensor],
            variable: List[str],
            stochastic: bool = False    
        ) -> None:
            raise NotImplementedError()

        
    def __init__(self) -> None:
        self.stochastic_predictions: List[VALUE.Variable] = []
        self.deterministic_predictions: List[VALUE.Variable] = []


    def add_prediction(self, prediction: List[torch.Tensor]) -> None:
        raise NotImplementedError()

    def evaluate(self) -> None:
        raise NotImplementedError()
    
    def _lag_1_autocorrelation(self) -> float:
        raise NotImplementedError()

    def _median_of_the_annual_cold(self) -> float:
        raise NotImplementedError()

    def _median_of_the_annual_dry(self) -> float:
        raise NotImplementedError()

    def _relative_frequency_of_days_plus_20_deg(self) -> float:
        raise NotImplementedError()

    def _relative_frequency_of_days_plus_25_deg(self) -> float:
        raise NotImplementedError()

    def _relative_frequency_of_days_minus_0_deg(self) -> float:
        raise NotImplementedError()

    def _mean(self) -> float:
        raise NotImplementedError()

    def _2nd_percentile(self) -> float:
        raise NotImplementedError()

    def _98th_percentile(self) -> float:
        raise NotImplementedError()

    def _mean_wet_day_precipitation(self) -> float:
        raise NotImplementedError()

    def _median_of_the_annual_warm(self) -> float:
        raise NotImplementedError()

    def _median_of_the_annual_wet(self) -> float:
        raise NotImplementedError()

    def _kolmogorov_smirnov(self) -> float:
        raise NotImplementedError()

    def _pearson_correlation(self) -> float:
        raise NotImplementedError()

    def _ration_of_the_standard_deviations(self) -> float:
        raise NotImplementedError()

    def _root_mean_squre_error(self) -> float:
        raise NotImplementedError()

    def _spearman_correlation(self) -> float:
        raise NotImplementedError()