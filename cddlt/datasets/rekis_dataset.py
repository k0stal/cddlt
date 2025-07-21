import torch

from typing import TypedDict, Set, Tuple, Self, List
from pathlib import Path
from cddlt.datasets.netcdf_dataset import NetCDFDataset

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
    DATASET_NAME: str = "ReKIS"

    """The type of a single dataset element."""
    class Element(TypedDict):
        input: torch.Tensor
        target: torch.Tensor

    class Dataset(NetCDFDataset):
        def __init__(self, data_path: str, interval: Tuple[int, int], variables: List[str]) -> None:
            super().__init__(data_path, interval, variables)

        def __getitem__(self, index: int) -> "ReKIS.Element":
            item = super().__getitem__(index)
            return item, item

        def __len__(self) -> int:
            return super().__len__()
        
        def select(self, variables: List[str]) -> Self:
            self().reset_variables(variables)
            return self
        
        def _upscale(self, item: torch.Tensor) -> torch.Tensor:
            # tmp
            upscaled = torch.nn.functional.interpolate(
                item.unsqueeze(0),
                scale_factor=ReKIS.UPSCALE_FACTOR ** -1,
                mode="bicubic",
                antialias=True,
            )
            return upscaled.squeeze(0)

    def __init__(
        self, 
        *,
        data_path: str,
        variables: List[str],
        train_len: Tuple[int, int],
        dev_len: Tuple[int, int],
        test_len: Tuple[int, int]
    ) -> None:
        
        assert all(var in self.AVAILABLE_VARIABLES for var in variables), f"Selected variables are not supported, select from: {self.AVAILABLE_VARIABLES}"
        self.variables = variables

        intervals = [train_len, dev_len, test_len]
        for interval in intervals:
            assert interval[0] >= self.AVILABLE_YEARS[0] and interval[1] <= self.AVILABLE_YEARS[1], f"Selected years must be a subset of: {self.AVILABLE_YEARS}"    

        for dataset, interval in zip(self.SETS_NAMES, intervals):
            datset_path = Path(data_path) / self.DATASET_NAME
            setattr(self, dataset, self.Dataset(datset_path, interval, self.variables))

        print(f"ReKIS dataset initalized.\ntrain size: ({len(self.train)})\ndev size: ({len(self.dev)})")

    """Train dataset."""
    train: Dataset

    """Eval dataset."""
    dev: Dataset

    """Test dataset."""
    test: Dataset
