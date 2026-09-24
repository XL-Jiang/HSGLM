from omegaconf import DictConfig
from .abide import load_abide2_data,load_adhd_data
from .dataloader import init_stratified_kfold_dataloader,init_loso_dataloader
from typing import List
import torch.utils as utils


def dataset_factory(cfg: DictConfig,SEED) -> List[utils.data.DataLoader]:

    datasets = eval(
        f"load_{cfg.dataset.name}_data")(cfg)
    if cfg.cv_mode=='loso':
        dataloaders = init_loso_dataloader(cfg, *datasets,SEED)
    elif cfg.dataset.stratified:
        dataloaders = init_stratified_kfold_dataloader(cfg, *datasets,SEED)
    return dataloaders
