import torch

from typing import TypedDict, Set
from pathlib import Path
from dl_tools.datasets.netcdf_dataset import NetCDFDataset

"""
TODO: 
    - fix implementation.
"""

class CORDEX:

    """Supported file formats."""
    FILE_FORMATS: Set[str] = {"NetCDF"}

    """Supported variables."""
    VARIABLES: Set[str] = {"PR", "TM", "TN", "TX"}

    """Subsets of the CORDEX dataset."""
    SETS_NAMES: Set[str] = {"test"}
    
    """Datasets of the CORDEX dataset."""
    DATASET_NAMES: Set[str] = {"CORDEX_test"}

    """The type of a single dataset element."""
    class Element(TypedDict):
        input: torch.Tensor
        target: torch.Tensor

    class Datset(NetCDFDataset):
        def __init__(self, data_path: str) -> None:
            super().__init__(data_path)

        def __getitem__(self, index: int) -> "CORDEX.Element":
            item = self().__getitem__(index)
            return item, item

        def __len__(self) -> int:
            return super().__len__()

    def __init__(
        self, 
        *,
        data_path: str,
        channels: int,
    ) -> None:
        """Initialize the CORDEX datset."""
        self.channels = channels

        for dataset in self.DATASET_NAMES:
            datset_path = Path(data_path) / dataset
            setattr(self, dataset, self.Datset(datset_path))

        print(f"CORDEX dataset initalized.\ntest size: ({len(self.test)})")

    """Eval dataset."""
    test : Datset
