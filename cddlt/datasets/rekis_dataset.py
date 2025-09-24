import torch
import rasterio
import xarray as xr
import pandas as pd

from typing import TypedDict, Set, Tuple, List, Callable
from pathlib import Path
from cddlt.datasets.netcdf_dataset import NetCDFDataset
from cddlt.datasets.netcdf_residual_dataset import NetCDFResidualDataset

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
    RANGE_AVAILABLE: Tuple[str] = ("1961-01-01", "2024-01-01")

    """Subsets of the ERA5_ReKIS dataset."""
    SETS_NAMES: List[str] = ["train", "dev", "test"]
    
    """Datasets of the ERA5_ReKIS dataset."""
    DATASET_NAME: str = "ReKIS"

    """Transformation function."""    
    @staticmethod
    def REPROJECT_FN(method: str, residual: bool) -> Callable[[xr.DataArray], xr.DataArray]:
        try:
            res_method = getattr(rasterio.enums.Resampling, method)
        except AttributeError:
            raise ValueError(f"{method} is not a valid resampling method.")
        
        def reproject_fn(data: xr.DataArray) -> xr.DataArray: ### possibility of paralelization: num_threads arg
            
            ### reproject to match 
            repro_data = data.rio.reproject(
                data.rio.crs,
                resolution=(10_000, 10_000),
                resampling=res_method
            )
            
            ### iterpolate the input to match target resolution
            if residual: 
                repro_data = repro_data.rio.reproject(
                    repro_data.rio.crs,
                    resolution=(100_000, 100_000), ## original data resolution
                    resampling=res_method
                )
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
        fine: torch.Tensor

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
            element: ReKIS.Element = {
                "input": input,
                "target": target
            }
            return element

        def __len__(self) -> int:
            return super().__len__()
        
    class ResidualDataset(NetCDFResidualDataset):
        def __init__(
            self, 
            data_path: str, 
            interval: Tuple[str, str], 
            variables: List[str],
        ) -> None:
            super().__init__(data_path, interval, variables)

        def __getitem__(self, index: int) -> "ReKIS.ResidualElement":
            input, target, coarse, fine = super().__getitem__(index)
            element: ReKIS.Element = {
                "input": input,
                "target": target,
                "coarse": coarse,
                "fine": fine
            }
            return element

        def __len__() -> int:
            return super().__len__()

    def __init__(
        self, 
        *,
        data_path: str,
        variables: List[str],
        train_len: Tuple[str, str],
        dev_len: Tuple[str, str],
        test_len: Tuple[str, str],
        resampling: str = "cubic_spline",
        residual: bool = False,
        standardize: bool = True    
    ) -> None:
        
        assert all(var in self.AVAILABLE_VARIABLES for var in variables), f"Selected variables are not supported, select from: {self.AVAILABLE_VARIABLES}"
        self.variables = variables
        self.standard_params = {}

        intervals = [train_len, dev_len, test_len]
        for start, end in intervals:
            assert self._cmp_time_str(start, self.RANGE_AVAILABLE[0]), \
                f"Start date {start} must be >= {self.RANGE_AVAILABLE[0]}"
            assert self._cmp_time_str(self.RANGE_AVAILABLE[1], end), \
                f"End date {end} must be <= {self.RANGE_AVAILABLE[1]}"

        for dataset, interval in zip(self.SETS_NAMES, intervals):
            dataset_obj = self.Dataset(data_path, interval, self.variables) if not residual else self.ResidualDataset(data_path, interval, self.variables)
            if standardize:
                if dataset == "train":
                    self.standard_params = dataset_obj.fit_transform_std()
                else:
                    dataset_obj.transform_std(self.standard_params)
            dataset_obj.reproject(self.REPROJECT_FN(resampling, residual)) ### the order of std and reproject should be reversed to work with ResidualDataset
            dataset_obj.convert_to_tensors()
            setattr(self, dataset, dataset_obj)

        print(f"\nReKIS dataset initalized.\ntrain size: ({len(self.train)})\ndev size: ({len(self.dev)})\ntest size: ({len(self.test)})\n")

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

    """Train dataset."""
    train: Dataset

    """Eval dataset."""
    dev: Dataset

    """Test dataset."""
    test: Dataset
