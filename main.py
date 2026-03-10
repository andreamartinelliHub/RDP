# from src.RDP.registry import Registry, MODEL_REGISTRY
# import src.RDP.models  # <--- This trigger the RDP

from RDP import models, datasets, Registry, MODELS_REGISTRY, DATASETS_REGISTRY

import hydra
from omegaconf import DictConfig, OmegaConf

# def fetch_trained_model(experiment_path):
#     exp_dir = Path(experiment_path)
#     config_path = exp_dir / "config.yaml"
    
#     # 1. Find the best checkpoint (assuming standard Lightning naming)
#     ckpt_path = list((exp_dir / "checkpoints").glob("*.ckpt"))[0]

#     # 2. Load the config that was used DURING training
#     cfg = OmegaConf.load(config_path)

#     # 3. Re-instantiate the model class from the registry via Hydra
#     # We use the 'model' sub-config from the saved file
#     model = hydra.utils.instantiate(cfg.model)
    
#     # 4. Load the trained weights
#     import torch
#     checkpoint = torch.load(ckpt_path, map_location="cpu")
#     model.load_state_dict(checkpoint["state_dict"])
    
#     model.eval()
#     return model, cfg

# def compare_experiments(exp_paths, test_dataset_id):
#     # 1. Setup the shared dataset
#     dm_class = DATASETS_REGISTRY.get(test_dataset_id)
#     datamodule = dm_class(batch_size=32)
#     datamodule.setup(stage="test")
#     test_loader = datamodule.test_dataloader()

#     results = {}

#     for path in exp_paths:
#         # 2. Fetch model and the config it was born with
#         model, cfg = fetch_trained_model(path)
        
#         # 3. Run evaluation (e.g., using Lightning Trainer or raw Loop)
#         # metrics = trainer.test(model, dataloaders=test_loader)
#         # results[path.name] = metrics
        
#     return results

# @hydra.main(version_base=None, config_path="conf", config_name="config")
@hydra.main(version_base=None)
def main(conf: DictConfig) -> None:
    print(MODELS_REGISTRY)
    print(DATASETS_REGISTRY)

    Registry.show_all()

if __name__ == "__main__":
    main()
