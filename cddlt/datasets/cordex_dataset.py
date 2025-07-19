import torch

from typing import TypedDict, Set, Tuple, List, Self
from pathlib import Path
from cddlt.datasets.netcdf_dataset import NetCDFDataset

class CORDEX:

    """Supported file formats"""
    FILE_FORMATS: Set[str] = {"NetCDF"}

    """Supported variables."""
    AVAILABLE_VARIABLES: Set[str] = {"PR", "TM", "TN", "TX"}

    """Data range."""
    AVILABLE_YEARS: Tuple[int] = (1979, 2023)

    """Subsets of the CORDEX dataset."""
    SETS_NAMES: Set[str] = {"dev", "test"}
    
    """Datasets of the CORDEX dataset."""
    DATASET_NAME: str = "CORDEX"

    """The type of a single dataset element."""
    class Element(TypedDict):
        input: torch.Tensor
        target: torch.Tensor

    class Dataset(NetCDFDataset):
        def __init__(self, data_path: str, interval: Tuple[int, int], variables: List[str]) -> None:
            super().__init__(data_path, interval, variables)

        def __getitem__(self, index: int) -> "CORDEX.Element":
            item = super().__getitem__(index)
            return item, item

        def __len__(self) -> int:
            return super().__len__()
        
        def select(self, variables: List[str]) -> Self:
            self().reset_variables(variables)
            return self

    def __init__(
        self, 
        *,
        data_path: str,
        variables: List[str],
        dev_len: Tuple[int, int],
        test_len: Tuple[int, int]
    ) -> None:
        
        assert all(var in self.AVAILABLE_VARIABLES for var in variables), f"Selected variables are not supported, select from: {self.AVAILABLE_VARIABLES}"
        self.variables = variables

        intervals = [dev_len, test_len]
        for interval in intervals:
            assert interval[0] >= self.AVILABLE_YEARS[0] and interval[1] <= self.AVILABLE_YEARS[1], f"Selected years must be a subset of: {self.AVILABLE_YEARS}"    

        for dataset, interval in zip(self.SETS_NAMES, intervals):
            datset_path = Path(data_path) / self.DATASET_NAME
            setattr(self, dataset, self.Dataset(datset_path, interval, self.variables))

        print(f"CORDEX dataset initalized.\ndev size: ({len(self.dev)})\ntest size: ({len(self.test)})")


    """Eval dataset."""
    dev: Dataset

    """Test dataset."""
    test: Dataset
