
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import hydra
import traceback

# 
from RDP import models, datasets, utils, Registry, MODELS_REGISTRY, DATASETS_REGISTRY
import logging
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='src/RDP/configs', config_name='base_config')
def train(conf: DictConfig) -> None:
    
    ### CHECK VALID IMPLEMENTATIONS:
    logger.info(MODELS_REGISTRY)
    logger.info(DATASETS_REGISTRY)

    ### TS NAME
    logger.info(f">> TS NAME: {conf.ts.name}")
    logger.info(f">> Check your Root: {conf.user.paths.root}")
    dataset_folder, model_folder = utils.check_folder_path_init(conf.user)

    breakpoint()

    ### LOADING THE DATASET, creating a TS object
    dataset_class = DATASETS_REGISTRY.get(conf.flow.dataset_name)
    ds_class = dataset_class(conf.flow.dataset_version, dataset_folder)
    try:
        ts = ds_class.load(ts_name = conf.ts.name)
    except Exception:
        logger.info(f"LOADING {conf.flow.dataset_name}[v={conf.flow.dataset_version}] ERROR {traceback.format_exc()}")
        raise


    ### LOADING THE MODEL
    model_class = MODELS_REGISTRY.get(conf.flow.model_name)
    model_version = conf.flow.model_version
    try:
        model = model_class(model_version)
    except Exception:
        logger.info(f"LOADING {conf.flow.model_name} ERROR {traceback.format_exc()}")
        raise



    return ##for optuna!

if __name__ == '__main__': 
    train()