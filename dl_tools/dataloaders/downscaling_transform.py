import torch
from typing import Type, Dict, Union, Any, List, Tuple, Optional, Callable

"""
#TODO
    - add workers to dataloader method.
"""

class DownscalingTransform:
    def __init__(
        self, 
        dataset: Type[torch.utils.data.Dataset],
        transform: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
    ) -> None:
        self.dataset = dataset
        self.transform = transform if transform is not None else self.default_transform
        self.collate_fn = collate_fn if collate_fn is not None else self.default_collate_fn

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, Any]]:
        sample = self.dataset[index]
        return self.transform(sample)
    
    ## transform from [batch_size, channels, lat, lon] -> [batch_size, grid_size] 
    ## assuming we will always be modeling single variable?
    def default_transform(self, sample: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return sample['input'], sample['target']

    def default_collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        images = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        return images, targets

    def dataloader(self, batch_size: int, shuffle: bool = False) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )
