# customized imports
import pandas as pd
from pathlib import Path

# forced imports
from RDP.registry import DATASETS_REGISTRY
from RDP import TimeSeries
from ..utils import load_config
import logging
logger = logging.getLogger(__name__)


@DATASETS_REGISTRY.register("synthetic")
class SyntheticDataset:
    def __init__(self, version, folder_position: Path):
        """Each dataset has its own folder, containing: [data file(s), .py to handle  the loading, .yaml for the versioning]
        > .py reads data file(s) and cleans them
        """
        self.version = version
        self.folder_position = folder_position
        self.yaml_path = folder_position/'synthetic/synthetic.yaml' # (@^@)
        assert self.yaml_path.exists(), f"{self.__class__.__name__}.yaml_path does not exists!\nNow it is {self.yaml_path}"
        self.dataset_conf = load_config(config_path = self.yaml_path)[version]
        logger.info(f"Version {version} from yaml file {self.yaml_path}")
    
    def load(self, ts_name):        
        data = pd.read_csv(self.folder_position / str(self.dataset_conf.data_path))
        data['time'] = pd.to_datetime(data['time'])
        columns = [c for c in data.columns if c not in ['y','time']]

        ts = TimeSeries(ts_name)
        ts.load_signal(data,
                    target_vars = self.dataset_conf.get('target_vars', ['y']),
                    group = self.dataset_conf.get('group', None),
                    enrich_cat = self.dataset_conf.get('enrich_cat',None),
                    past_vars = columns if self.dataset_conf.get('use_past_variables',False) else [],
                    future_vars = self.dataset_conf.get('future_vars', []),
                    categorical_past_vars = self.dataset_conf.get('categorical_past_vars', []),
                    categorical_fut_vars = self.dataset_conf.get('categorical_fut_vars', []),
                    check_past = self.dataset_conf.get('check_past', []),
                    check_holes_and_duplicates = self.dataset_conf.get('check_holes_and_duplicates', []),
                    silly_model = self.dataset_conf.get('silly',False),
                    sampler_weights = self.dataset_conf.get('sampler_weights',None),
                    )
        logger.info(ts)
        return ts