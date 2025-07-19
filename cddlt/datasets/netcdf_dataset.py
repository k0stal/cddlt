import torch
import os
import xarray as xr
import numpy as np
from typing import List, Tuple, Self
from pathlib import Path
import pandas as pd

class NetCDFDataset(torch.utils.data.Dataset):
    SUFFIXES: List[str] = [".nc", ".nc4"]
    
    def __init__(self, data_path: str, interval: Tuple[int, int], variables: List[str]):
        super().__init__()
        assert os.path.isdir(data_path), f"{data_path} is not a valid directory."
        
        self.data_path = Path(data_path)
        self.start_year, self.end_year = interval
        self.variables = variables
        
        self.nc_files = self._get_nc_files()
        assert len(self.nc_files) > 0, f"No NetCDF files found in {data_path}"
        self._load_data()
        
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
        )
        
        if not dataset:
            raise RuntimeError("No datasets could be loaded successfully")
    
        time_mask = ((dataset.time.dt.year >= self.start_year) & (dataset.time.dt.year <= self.end_year))
        filtered_ds = dataset.sel(time=time_mask)
        
        if len(filtered_ds.time) == 0:
            raise ValueError(f"No data found for years {self.start_year}-{self.end_year}")
        
        self.dataset = filtered_ds
        self.time_coords = filtered_ds.time.values
        
        print(f"Loaded data shape: {dict(filtered_ds.dims)}")
        print(f"Time range: {pd.to_datetime(self.time_coords[0])} to {pd.to_datetime(self.time_coords[-1])}")
        print(f"Variables in dataset: {list(filtered_ds.data_vars.keys())}")
        
    def reset_variables(self, variables: List[str]) -> Self:
        self.variables = variables
        return self
    
    def __getitem__(self, index: int) -> torch.Tensor:

        if index < 0 or index >= len(self.time_coords):
            raise IndexError(f"Index {index} out of bounds.")
        
        time_slice = self.dataset.isel(time = index)
        time_slice = time_slice[self.variables]
        
        if len(time_slice.data_vars) == 1:
            var_name = list(time_slice.data_vars.keys())[0]
            data_array = time_slice[var_name].values
            if data_array.ndim == 2:
                data_array = data_array[np.newaxis, ...]
        else:
            arrays = []
            for var_name in time_slice.data_vars:
                var_array = time_slice[var_name].values
                arrays.append(var_array)
            data_array = np.stack(arrays, axis=0)
        
        return torch.tensor(data_array, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.time_coords)
