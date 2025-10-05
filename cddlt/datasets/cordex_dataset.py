import torch
import pandas as pd
import rasterio
import xarray as xr

from typing import TypedDict, Set, Tuple, List, Dict, Callable
from cddlt.datasets.netcdf_dataset import NetCDFDataset
from cddlt.datasets.netcdf_residual_dataset import NetCDFResidualDataset

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

    """Transformation function."""    
    @staticmethod
    def REPROJECT_FN(method: str) -> Callable[[xr.DataArray], xr.DataArray]:
        try:
            res_method = getattr(rasterio.enums.Resampling, method)
        except AttributeError:
            raise ValueError(f"{method} is not a valid resampling method.")
        
        def reproject_fn(data: xr.DataArray) -> xr.DataArray: ### possibility of paralelization: num_threads arg
            # reproject_match would be better?
            repro_data = data.rio.reproject(
                data.rio.crs,
                resolution=(1_000, 1_000),
                resampling=res_method
            )
            
            #repro_data = repro_data.rename({"x": "easting", "y": "northing"})
            return repro_data
        
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

    class ResidualElement(TypedDict):
        input: torch.Tensor
        taget: torch.Tensor
        coarse: torch.Tensor

    class BaseDataset(NetCDFDataset):
        """Abstract dataset wrapper for CORDEX"""

        def __init__(self, data_path: str, interval: Tuple[str, str], variables: List[str]):
            super().__init__(data_path, interval, variables)

        def __len__(self) -> int:
            return super().__len__()

    class Dataset(BaseDataset):
        """Regular dataset"""
        def __getitem__(self, index: int) -> "CORDEX.Element":
            input, target = super().__getitem__(index)
            return {"input": input, "target": target}

    class ResidualDataset(BaseDataset, NetCDFResidualDataset):
        """Residual dataset"""
        def __getitem__(self, index: int) -> "CORDEX.ResidualElement":
            input, target, coarse, fine = super().__getitem__(index)
            return {"input": input, "target": target, "coarse": coarse, "fine": fine}

    def __init__(
        self, 
        *,
        data_path: str,
        variables: List[str],
        dev_len: Tuple[str, str],
        test_len: Tuple[str, str],
        standardize: bool = True,
        standard_params: Dict[str, Dict[str, float]] = {},   # we will use the fitted params from rekis
        resampling_method: str = "cubic_spline",
        residual: bool = False
    ) -> None:
        
        assert all(var in self.AVAILABLE_VARIABLES for var in variables), f"Selected variables are not supported, select from: {self.AVAILABLE_VARIABLES}"
        if standardize:
            assert standard_params != {}, f"Standardization without specified params."

        self.variables = variables
        self.standardize = standardize
        self.standard_params = standard_params
        self.resampling_method = resampling_method

        intervals = [dev_len, test_len]
        for start, end in intervals:
            assert self._cmp_time_str(start, self.RANGE_AVAILABLE[0]), \
                f"Start date {start} must be >= {self.RANGE_AVAILABLE[0]}"
            assert self._cmp_time_str(self.RANGE_AVAILABLE[1], end), \
                f"End date {end} must be <= {self.RANGE_AVAILABLE[1]}"

        for dataset, interval in zip(self.SETS_NAMES, intervals):
            dataset_obj = self._make_dataset(residual, data_path, interval, self.variables)
            if self.standardize:
                dataset_obj.transform_std(self.standard_params)
            dataset_obj.convert_to_tensors()
            setattr(self, dataset, dataset_obj)

        print(f"\nCORDEX dataset initalized.\ndev size: ({len(self.dev)})\ntest size: ({len(self.test)})\n")

    def _make_dataset(self, residual: bool, data_path: str, interval: Tuple[str, str], variables: List[str]) -> Dataset | ResidualDataset:
        ds_class = self.ResidualDataset if residual else self.Dataset
        ds = ds_class(data_path, interval, variables)
        if residual: ds.reproject(self.REPROJECT_FN(self.resampling_method))
        return ds

    def destandardize(self, input: List[torch.Tensor]) -> List[torch.Tensor]:
        _, C, _, _ = input[0].shape
        assert C == len(self.variables), f"Expected {len(self.variables)} variables, got {C}."

        output = []
        for tensor in input:
            destandardized_channels = []
            for idx, variable in enumerate(self.variables):
                std = torch.tensor(self.standard_params[variable]["std"], dtype=tensor.dtype, device=tensor.device)
                mean = torch.tensor(self.standard_params[variable]["mean"], dtype=tensor.dtype, device=tensor.device)
                channel = tensor[:, idx, :, :] * std + mean
                destandardized_channels.append(channel.unsqueeze(1))

            destandardized_tensor = torch.cat(destandardized_channels, dim=1)
            output.append(destandardized_tensor)

        return output

    """Eval dataset."""
    dev: Dataset

    """Test dataset."""
    test: Dataset
