from .training_monitoring import training_monitoring
from omegaconf import DictConfig
from typing import List
import torch
import logging
import torch.utils.data as utils


def training_factory(config: DictConfig,
                     model: torch.nn.Module,
                     optimizers: List[torch.optim.Optimizer],
                     dataloaders: List[utils.DataLoader],
                     logger: logging.Logger):

    train = config.model.get("train", None)
    if not train:
        train = config.training.name
    return eval(train)(cfg=config,
                       model=model,
                       optimizers=optimizers,
                       dataloaders=dataloaders,
                       logger=logger)
