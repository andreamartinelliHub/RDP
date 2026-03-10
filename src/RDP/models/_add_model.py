from pathlib import Path
import argparse
import sys

# Blueprint
MODEL_TEMPLATE = """import torch
from torch import nn
{import_layers}

try:
    import lightning.pytorch as pl
    from .base_v2 import Base
    OLD_PL = False
except:
    import pytorch_lightning as pl
    OLD_PL = True
    from .base import Base

from .utils import get_scope
from ..registry import MODELS_REGISTRY

@MODELS_REGISTRY.register("{model_id}")
class {model_name}(Base):
    # change according to what the model can do:
    handle_multivariate = False
    handle_future_covariates = False
    handle_categorical_variables = False
    handle_quantile_loss = False
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)

    def __init__(self, 
                 d_model = 128, 
                 # and other parameters,
                 **kwargs) -> None:
        super().__init__()

        # self.layer = nn.Linear...

    def can_be_compiled(self):
        return False
    
    def forward(self, batch):
        
        x = batch['x_num_past']
        batch_size = x.shape[0]
        # ... compute output

        return # output # shape = [batch_size, future_length, channels, quantiles]
"""

LAYER_TEMPLATE = """
import torch
from torch import nn

class CustomLayer(nn.Module):
    def __init__(self,
            in_dim: int,
            out_dim: int,
        ) -> None:
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer(x)
        return out
"""

def create_model_file(model, needed_aux_folder=True):
    
    model_id = model.lower()
    model_name = model_id.capitalize()
    
    model_file_path = Path(f"{model_name}.py")
    aux_folder = Path(f"./{model_id}/")

    # check possible overwriting
    ovewrite_smth = False
    if model_file_path.exists():
        print(f"🛑 '{model_file_path}' already exists!")
        ovewrite_smth = True
    if (needed_aux_folder and aux_folder.exists()):
        print(f"🛑 '{aux_folder}/' already exists!")
        ovewrite_smth = True
        
    if ovewrite_smth:
        print("Overwriting something! Maybe you should change the model name!")
        response = input("Do you want to overwrite it anyway? (Do you want to delete the existing file?) [yes/N]:")
        # only if you write exactly 'yes' you'll overwrite it!
        if response != 'yes':
            sys.exit(1)

    import_layers = f'from .{aux_folder}.layers import CustomLayer' if needed_aux_folder else ''
    # Formatting the template
    formatted_code = MODEL_TEMPLATE.format(
                        model_name = model_name,
                        model_id = model_id,
                        import_layers = import_layers
                    )
    model_file_path.write_text(formatted_code)
    print(f"✅ Created class {model_name} in {model_file_path}")
    
    if needed_aux_folder:
        aux_folder.mkdir(parents=True, exist_ok=True)
        (aux_folder / "__init__.py").touch()
        (aux_folder / "layers.py").write_text(LAYER_TEMPLATE)
        ### Add a config ???
        print("✅ Created auxiliar folder for custom layers")


def main():
    
    parser = argparse.ArgumentParser(
        description="Generate model files and folders. This file has to be located in the folder src/RDP/model."
        )
    parser.add_argument("model", type=str, help="The name of the model class to create")
    parser.add_argument("-a","-l","-c","--aux", action="store_true", help="Create an auxiliary layers folder")
    args = parser.parse_args()

    # path_models_folder = '/home/anmartinelli/FBK/RDP/src/RDP/models' # to-do: avoid hardcoded

    create_model_file(args.model, args.aux)


if __name__ == "__main__":
    main()