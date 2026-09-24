from omegaconf import DictConfig
from .HSGLM import HSGLM
def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    return eval(config.model.name)(config).cuda()
