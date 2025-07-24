import torch
import pandas as pd

from typing import TypedDict, Set, Tuple, List
from cddlt.datasets.netcdf_dataset import NetCDFDataset

class CORDEX:

    """Supported file formats"""
    FILE_FORMATS: Set[str] = {"NetCDF"}

    """Supported variables."""
    AVAILABLE_VARIABLES: Set[str] = {"PR", "TM", "TN", "TX"}

    """Data range."""
    RANGE_AVAILABLE: Tuple[str] = ("1979-01-02", "2013-01-01")

    """Subsets of the CORDEX dataset."""
    SETS_NAMES: Set[str] = ["dev", "test"]
    
    """Datasets of the CORDEX dataset."""
    DATASET_NAME: str = "CORDEX"
    
    @staticmethod
    def _cmp_time_str(a: str, b: str) -> bool:
        try:
            return pd.to_datetime(a) >= pd.to_datetime(b)
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}. Expected format is {format}.") from e


    """The type of a single dataset element."""
    class Element(TypedDict):
        input: torch.Tensor
        target: torch.Tensor

    class Dataset(NetCDFDataset):
        def __init__(self, data_path: str, interval: Tuple[str, str], variables: List[str]) -> None:
            super().__init__(data_path, interval, variables)

        def __getitem__(self, index: int) -> "CORDEX.Element":
            input, target = super().__getitem__(index)
            return input, target

        def __len__(self) -> int:
            return super().__len__()

    def __init__(
        self, 
        *,
        data_path: str,
        variables: List[str],
        dev_len: Tuple[str, str],
        test_len: Tuple[str, str],
        #resampling: str = "cubic_spline"   # data already prepared
    ) -> None:
        
        assert all(var in self.AVAILABLE_VARIABLES for var in variables), f"Selected variables are not supported, select from: {self.AVAILABLE_VARIABLES}"
        self.variables = variables

        intervals = [dev_len, test_len]
        for start, end in intervals:
            assert self._cmp_time_str(start, self.RANGE_AVAILABLE[0]), \
                f"Start date {start} must be >= {self.RANGE_AVAILABLE[0]}"
            assert self._cmp_time_str(self.RANGE_AVAILABLE[1], end), \
                f"End date {end} must be <= {self.RANGE_AVAILABLE[1]}"

        for dataset, interval in zip(self.SETS_NAMES, intervals):
            #dataset_path = Path(data_path) / self.DATASET_NAME
            dataset_obj = self.Dataset(data_path, interval, self.variables)
            dataset_obj.convert_to_tensors()
            setattr(self, dataset, dataset_obj)

        print(f"\nCORDEX dataset initalized.\ndev size: ({len(self.dev)})\ntest size: ({len(self.test)})\n")


    """Eval dataset."""
    dev: Dataset

    """Test dataset."""
    test: Dataset
