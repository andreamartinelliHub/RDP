from ..registry import DATASETS_REGISTRY
import lightning.pytorch as pl

@DATASETS_REGISTRY.register("random")
class RandomDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=32):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        pass

    def load_data(self):
        pass