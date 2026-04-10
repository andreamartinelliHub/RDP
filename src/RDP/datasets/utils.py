from omegaconf import OmegaConf
from pathlib import Path

#
# import logging
# logger = logging.getLogger(__name__)

def load_config(config_path: Path):
    assert config_path.exists(), f"No config found at {config_path}"
    # logger.info(f"Dataset yaml loaded: {config_path}")
    conf = OmegaConf.load(config_path)
    return conf
