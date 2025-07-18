import torch

from typing import TypedDict, Set, Tuple, Self, List
from pathlib import Path
from dl_tools.datasets.netcdf_dataset import NetCDFDataset

"""
TODO: 
    - Maybe add two list of variables (not SuperResolution approach):
        1) variables to predict from
    - potentially include other file formats.
"""

class ReKIS:

    """Supported file formats"""
    FILE_FORMATS: Set[str] = {"NetCDF"}

    """Supported variables."""
    AVAILABLE_VARIABLES: Set[str] = {"PR", "TM", "TN", "TX"}

    """Data range."""
    AVILABLE_YEARS: Tuple[int] = (1979, 2023)

    """Subsets of the ERA5_ReKIS dataset."""
    SETS_NAMES: Set[str] = {"train", "dev"}
    
    """Datasets of the ERA5_ReKIS dataset."""
    DATASET_NAMES: Set[str] = {"ReKIS"}

    """Upscaling factor to match CORDEX"""
    UPSCALE_FACTOR: int = ...

    """The type of a single dataset element."""
    class Element(TypedDict):
        input: torch.Tensor
        target: torch.Tensor

    class Datset(NetCDFDataset):
        def __init__(self, data_path: str, interval: Tuple[int, int], variables: List[str]) -> None:
            super().__init__(data_path, interval, variables)

        def __getitem__(self, index: int) -> "ReKIS.Element":
            item = self().__getitem__(index)
            return self._upscale(item), item

        def __len__(self) -> int:
            return super().__len__()
        
        def select(self, variables: List[str]) -> Self:
            self().reset_variables(variables)
            return self
        
        def _upscale(self, item: torch.Tensor) -> torch.Tensor:
            upscaled = torch.nn.functional.interpolate(
                item,
                scale_factor=self.UPSCALE_FACTOR ** -1,
                mode="bicubic",
                antialias=True,
            )
            return upscaled

    def __init__(
        self, 
        *,
        data_path: str,
        variables: List[str],
        train_len: Tuple[int, int],
        dev_len: Tuple[int, int],
    ) -> None:
        
        assert variables in self.VARIABLES, f"Selected variables are not supported, select from: {self.AVAILABLE_VARIABLES.keys}"
        self.variables = variables

        intervals = [train_len, dev_len]
        for interval in intervals:
            assert interval[0] >= self.AVILABLE_YEARS and interval[1] <= self.AVILABLE_YEARS, f"Selected years must be a subset of: {self.AVILABLE_YEARS}"    

        for dataset, interval in zip(self.SETS_NAMES, intervals):
            datset_path = Path(data_path) / self.DATASET_NAMES[0]
            setattr(self, dataset, self.Datset(datset_path, self.variables, interval))

        print(f"CORDEX_ReKIS dataset initalized.\ntrain size: ({len(self.train)})\ndev size: ({len(self.dev)})")

    """Train dataset."""
    train: Datset

    """Eval dataset."""
    dev: Datset
