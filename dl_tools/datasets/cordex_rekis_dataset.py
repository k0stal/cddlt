import torch

from typing import TypedDict, Set, Tuple, Self, List
from pathlib import Path
from dl_tools.datasets.netcdf_dataset import NetCDFDataset

"""
TODO: 
    - potentially include other file formats.
"""

class CORDEX_ReKIS:

    """Supported file formats"""
    FILE_FORMATS: Set[str] = {"NetCDF"}

    """Supported variables."""
    AVAILABLE_VARIABLES: Set[str] = {"PR", "TM", "TN", "TX"}

    """Data range."""
    AVILABLE_YEARS: Tuple[int] = (1979, 2012)

    """Subsets of the CORDEX_ReKIS dataset."""
    SETS_NAMES: Set[str] = {"test"}
    
    """Datasets of the CORDEX_ReKIS dataset."""
    DATASET_NAMES: Set[str] = {"CORDEX", "ReKIS"}

    """The type of a single dataset element."""
    class Element(TypedDict):
        input: torch.Tensor
        target: torch.Tensor

    class CombinedDataset:
        def __init__(self, cordex_dataset: str, rekis_path: str, interval: Tuple[int, int], variables: List[str]) -> None:
            self.cordex_dataset = NetCDFDataset(cordex_dataset, interval, variables)
            self.rekis_dataset = NetCDFDataset(rekis_path, interval, variables)

            assert len(self.cordex_dataset) is len(self.rekis_dataset), f"Datasets have different lengths: era5({len(self.era5_datset)}), rekis({len(self.rekis_dataset)})"

        def __getitem__(self, index: int) -> "CORDEX_ReKIS.Element":
            return self.cordex_dataset[index], self.rekis_dataset[index]

        def __len__(self) -> int:
            return len(self.cordex_dataset)
        
        def select(self, variables: List[str]) -> Self:
            self.cordex_dataset.reset_variables(variables)
            self.rekis_dataset.reset_variables(variables)

    def __init__(
        self, 
        *,
        data_path: str,
        dev_len: Tuple[int, int],
        variables: List[str]
    ) -> None:
        
        assert variables in self.VARIABLES, f"Selected variables are not supported, select from: {self.AVAILABLE_VARIABLES.keys}"
        self.variables = variables

        assert dev_len[0] >= self.AVILABLE_YEARS and dev_len[1] <= self.AVILABLE_YEARS, f"Selected years must be a subset of: {self.AVILABLE_YEARS}"   

        for dataset in self.SETS_NAMES:
            dataset_paths = [Path(data_path) / subset for subset in self.DATASET_NAMES ]
            setattr(self, dataset, self.CombinedDataset(*dataset_paths, dev_len, self.variables))

        print(f"CORDEX_ReKIS dataset initalized.\ndev size: ({len(self.dev)})")

    """Eval dataset."""
    test: CombinedDataset
