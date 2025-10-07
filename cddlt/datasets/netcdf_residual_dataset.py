import torch
import xarray as xr

from typing import List, Tuple, Callable, Dict

from cddlt.datasets.netcdf_dataset import NetCDFDataset

class NetCDFResidualDataset(NetCDFDataset):
    def __init__(
        self, 
        data_path: str, 
        interval: Tuple[str, str], 
        variables: List[str], 
    ) -> None:
        super().__init__(data_path, interval, variables)
        self.coarse = self.dataset
        self.fine = self.dataset

    def fit_transform_std(self) -> Dict[str, Dict[str, float]]:
        params = {}

        for attr_name in ["inputs", "targets"]:
            attr_params = {}
            for var_name in sorted(self.dataset.data_vars):
                var = getattr(self, attr_name)[var_name]
                mean = var.mean().values
                std = var.std().values
                var_params = {"mean": mean, "std": std}
                attr_params[var_name] = var_params

                std_variable = (var - mean) / std
                getattr(self, attr_name)[var_name] = std_variable
            params[attr_name] = attr_params

        return params

    def transform_std(self, params: Dict[str, Dict[str, float]]) -> None:
        assert len(params) == 2, f"Directory should contain parameters for both inputs and targets."
        assert len(self.dataset.data_vars) == len(params["inputs"]) and \
               len(self.dataset.data_vars) == len(params["targets"]), \
            f"Dictionary should contain {len(self.dataset.data_vars)} variables."

        for attr_name in ["inputs", "targets"]:
            attr = getattr(self, attr_name)
            for var_name in sorted(attr.data_vars):
                mean = params[attr_name][var_name]["mean"]
                std = params[attr_name][var_name]["std"]
                std_variable = (attr[var_name] - mean) / std
                attr[var_name] = std_variable
            setattr(self, attr_name, attr)
    
    def reproject(
        self,
        reproject_fn: Callable[[xr.DataArray], xr.DataArray],
    ) -> None:
        inputs = reproject_fn(self.dataset)
        targets = self.dataset - inputs

        self.inputs = inputs
        self.targets = targets
        self.coarse = inputs.copy()
        self.fine = self.dataset

    def convert_to_tensors(self) -> None:
        for attr_name in ["inputs", "targets", "coarse", "fine"]:
            attr = getattr(self, attr_name)
            variables = []
            for var_name in sorted(attr.data_vars):
                tensor = torch.from_numpy(attr[var_name].values).unsqueeze(1)
                variables.append(tensor)

            tensor_data = torch.cat(variables, dim=1)
            setattr(self, attr_name, tensor_data) 
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input = self.inputs[index]
        target = self.targets[index]
        coarse = self.coarse[index]
        fine = self.fine[index]

        return input, target, coarse, fine
