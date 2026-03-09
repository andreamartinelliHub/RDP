
def get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss):
    message = f'Can {"NOT" if not handle_multivariate else "" }  handle multivariate output \n'\
                  f'Can {"NOT" if not handle_future_covariates else "" }  handle future covariates\n'\
                  f'Can {"NOT" if not handle_categorical_variables else "" }  handle categorical covariates\n'\
                  f'Can {"NOT" if not handle_quantile_loss else "" }  handle Quantile loss function'
            
    return message