import torch
import rasterio
import xarray as xr
import pandas as pd

from typing import TypedDict, Set, Tuple, Self, List, Callable
from pathlib import Path
from cddlt.datasets.netcdf_dataset import NetCDFDataset

"""
TODO:
use exact dates from datetime instead of years.
"""

class ReKIS:

    """Supported file formats"""
    FILE_FORMATS: Set[str] = {"NetCDF"}

    """Supported variables."""
    AVAILABLE_VARIABLES: Set[str] = {"PR", "TM", "TN", "TX"}

    """Data range."""
    RANGE_AVAILABLE: Tuple[str] = ("1979-01-01", "2023-12-31")

    """Subsets of the ERA5_ReKIS dataset."""
    SETS_NAMES: List[str] = ["train", "dev", "test"]
    
    """Datasets of the ERA5_ReKIS dataset."""
    DATASET_NAME: str = "ReKIS"

    """Transformation function."""    
    @staticmethod
    def REPROJECT_FN(method: str) -> Callable[[xr.DataArray], xr.DataArray]:
        try:
            res_method = getattr(rasterio.enums.Resampling, method)
        except AttributeError:
            raise ValueError(f"{method} is not a valid resampling method.")
        
        def reproject_fn(data: xr.DataArray) -> xr.DataArray:
            return data.rio.reproject(
                data.rio.crs,
                resolution=(10_000, 10_000),
                resampling=res_method
            )
        return reproject_fn
    
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
        def __init__(
            self, 
            data_path: str, 
            interval: Tuple[str, str], 
            variables: List[str],
        ) -> None:
            super().__init__(data_path, interval, variables)

        def __getitem__(self, index: int) -> "ReKIS.Element":
            input, target = super().__getitem__(index)
            return input, target

        def __len__(self) -> int:
            return super().__len__()

    def __init__(
        self, 
        *,
        data_path: str,
        variables: List[str],
        train_len: Tuple[str, str],
        dev_len: Tuple[str, str],
        test_len: Tuple[str, str],
        resampling: str = "cubic_spline"    
    ) -> None:
        
        assert all(var in self.AVAILABLE_VARIABLES for var in variables), f"Selected variables are not supported, select from: {self.AVAILABLE_VARIABLES}"
        self.variables = variables

        intervals = [train_len, dev_len, test_len]
        for start, end in intervals:
            assert self._cmp_time_str(start, self.RANGE_AVAILABLE[0]), \
                f"Start date {start} must be >= {self.RANGE_AVAILABLE[0]}"
            assert self._cmp_time_str(self.RANGE_AVAILABLE[1], end), \
                f"End date {end} must be <= {self.RANGE_AVAILABLE[1]}"

        for dataset, interval in zip(self.SETS_NAMES, intervals):
            #dataset_path = Path(data_path) / self.DATASET_NAME
            dataset_obj = self.Dataset(data_path, interval, self.variables)
            dataset_obj.slice_data()
            dataset_obj.set_spatial_dims()
            dataset_obj.reproject(self.REPROJECT_FN(resampling), split_target=True)
            dataset_obj.convert_to_tensors()
            setattr(self, dataset, dataset_obj)

        print(f"ReKIS dataset initalized.\ntrain size: ({len(self.train)})\ndev size: ({len(self.dev)})\ntest size: ({len(self.test)})")

    """Train dataset."""
    train: Dataset

    """Eval dataset."""
    dev: Dataset

    """Test dataset."""
    test: Dataset
