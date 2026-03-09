
from .utils import get_scope
from ..registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("new_model")
class New_model:
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
