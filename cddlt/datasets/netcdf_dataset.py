import torch
import os
import rasterio
import xarray as xr
import pandas as pd
import pandas as pd

from typing import List, Tuple, Callable, Set
from pathlib import Path

class NetCDFDataset(torch.utils.data.Dataset):
    SUFFIXES: List[str] = [".nc", ".nc4"]

    @staticmethod
    def _cnv_date(date: str) -> pd.Timestamp:
        try:
            return pd.to_datetime(date)
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}. Expected format is {format}.") from e
        
    @staticmethod
    def _get_resampling_enum(method: str) -> rasterio.enums.Resampling:
        try:
            return getattr(rasterio.enums.Resampling, method)
        except AttributeError:
            raise ValueError(f"{method} is not a valid resampling method.")
    
    def __init__(
        self, 
        data_path: str, 
        interval: Tuple[str, str], 
        variables: List[str], 
        chunks: dict = None
    ):
        super().__init__()
        assert os.path.isdir(data_path), f"{data_path} is not a valid directory."
        
        self.data_path = Path(data_path)
        start_date, end_date = interval
        self.start_date, self.end_date = self._cnv_date(start_date), self._cnv_date(end_date)
        self.variables = variables
        self.chunks = chunks or {'time': 50}
        
        self.nc_files = self._get_nc_files()
        assert len(self.nc_files) > 0, f"No NetCDF files found in {data_path}"
        self._load_data()

        self.inputs = self.dataset
        self.targets = self.dataset
  
    def reproject(
        self,
        reproject_fn: Callable[[xr.DataArray], xr.DataArray],
    ) -> None:
        self.dataset.rio.set_spatial_dims("easting", "northing")
        self.inputs = reproject_fn(self.dataset)
        self.targets = self.dataset

    def convert_to_tensors(self) -> None:
        for attr_name in ['inputs', 'targets']:
            attr = getattr(self, attr_name)
            variables = []
            for var_name in attr.data_vars:
                tensor = torch.from_numpy(attr[var_name].values).unsqueeze(1)
                variables.append(tensor)

            tensor_data = torch.cat(variables, dim=1)
            setattr(self, attr_name, tensor_data)
            
    def _get_nc_files(self) -> List[Path]:
        nc_files = []
        for file_path in self.data_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in self.SUFFIXES:
                nc_files.append(file_path)
        return sorted(nc_files)
    
    def _load_data(self):
        print(f"Loading {len(self.nc_files)} NetCDF file(s)...")

        dataset = xr.open_mfdataset(
            self.nc_files,
            decode_coords="all",
            chunks=self.chunks,
            parallel=True
        )
        
        if not dataset:
            raise RuntimeError("No datasets could be loaded successfully")
    
        # filter by both time and variables
        # both start and end dates are included
        time_mask = ((dataset.time >= self.start_date) & (dataset.time < self.end_date))
        filtered_ds = dataset.sel(time=time_mask)
        filtered_ds = filtered_ds[self.variables]
        
        if len(filtered_ds.time) == 0:
            raise ValueError(f"No data found for dates {self.start_date}-{self.end_date}")
        
        self.dataset = filtered_ds
        self.time_coords = filtered_ds.time.values
        
        print(f"Loaded data shape: {dict(filtered_ds.dims)}")
        print(f"Time range: {pd.to_datetime(self.time_coords[0])} to {pd.to_datetime(self.time_coords[-1])}")
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input = self.inputs[index]
        target = self.targets[index]

        return input, target
    
    def __len__(self) -> int:
        return len(self.time_coords)
