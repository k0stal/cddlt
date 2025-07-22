import torch
import affine
import rasterio
import pandas as pd
import xarray as xr

from typing import TypedDict, Set, Tuple, List, Self, Callable
from cddlt.datasets.netcdf_dataset import NetCDFDataset

class CORDEX:

    """Supported file formats"""
    FILE_FORMATS: Set[str] = {"NetCDF"}

    """Supported variables."""
    AVAILABLE_VARIABLES: Set[str] = {"PR", "TM", "TN", "TX"}

    """Temperature variables."""
    TEMP_VARIABLES: Set[str] = {"TM", "TN", "TX"}

    """Data range."""
    RANGE_AVAILABLE: Tuple[str] = ("1979-01-01", "2023-12-31")

    """Subsets of the CORDEX dataset."""
    SETS_NAMES: Set[str] = ["dev", "test"]
    
    """Datasets of the CORDEX dataset."""
    DATASET_NAME: str = "CORDEX"

    """Transformation function."""
    @staticmethod
    def REPROJECT_FN(method: str) -> Callable[[xr.DataArray], xr.DataArray]:
        try:
            res_method = getattr(rasterio.enums.Resampling, method)
        except AttributeError:
            raise ValueError(f"{method} is not a valid resampling method.")
        
        def reproject_fn(data: xr.DataArray) -> xr.DataArray:
            return data.rio.reproject(
                dst_crs="EPSG:31468",
                transform=affine.Affine(10000.0, 0.0, 4335000.0, 0.0, -10000.0, 5955000.0),
                shape=(40, 40),
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
        def __init__(self, data_path: str, interval: Tuple[str, str], variables: List[str]) -> None:
            super().__init__(data_path, interval, variables)

        def __getitem__(self, index: int) -> "CORDEX.Element":
            item = super().__getitem__(index)
            return item, item

        def __len__(self) -> int:
            return super().__len__()

    def __init__(
        self, 
        *,
        data_path: str,
        variables: List[str],
        dev_len: Tuple[str, str],
        test_len: Tuple[str, str],
        resampling: str = "cubic_spline"
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
            dataset_obj.convert_kelvin_to_celsius(self.TEMP_VARIABLES)
            dataset_obj.reproject(self.REPROJECT_FN(resampling), split_target=False)
            dataset_obj.convert_to_tensors()
            setattr(self, dataset, dataset_obj)

        print(f"CORDEX dataset initalized.\ndev size: ({len(self.dev)})\ntest size: ({len(self.test)})")


    """Eval dataset."""
    dev: Dataset

    """Test dataset."""
    test: Dataset
