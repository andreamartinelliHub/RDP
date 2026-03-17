class Registry:
    _all_registries = [] # variable to fetch instances

    def __init__(self, name):
        # Instance-level variables
        self._name = name
        self._module_dict = {}

        # to track created Registries
        Registry._all_registries.append(self)

    ### Functions for status checking
    def __repr__(self) -> str:
        # Version 0.0.1: printing a registry means check classes it is tracking 
        items = sorted(list(self._module_dict.keys()))
        return f"Registry(name='{self._name}', items={items})"
    
    @classmethod
    def show_all(cls):
        # print("\nShow each Registry!")
        print("\n--- GLOBAL REGISTRY STATUS ---")
        for registry in cls._all_registries:
            print(f"{registry._name:<10} | {', '.join(sorted(registry._module_dict.keys()))}")
        print("------------------------------\n")
    ### 

    ### Base Functions:
    # - register: save the class into the internal dict ['model name': model_class]
    # - get: from an external config use the string 'model name' to fetch the desired model_class
    # Use it as 
    # model_class = MODELS_REGISTRY.get(model_config["model_name"])
    # instance = model_class(model_config)
    def register(self, name=None, verbose = False):
        """Register the decorated class naming it 'name'. If name==None the class name will be used."""
        if verbose:
            print(f"Registry called for {name}") # just to track the code-flow

        def wrapper(module):
            # if name not provided, use the module name
            module_name = name if name is not None else module.__name__

            if module_name in self._module_dict:
                raise KeyError(f"{module_name} is already registered in {self._name}")
                
            # register the class in the registry
            self._module_dict[module_name] = module

            # A decorator must return the object it decorated
            return module
        return wrapper

    def get(self, name):
        """Fetch a class by its registered string name."""
        if name not in self._module_dict:
            raise KeyError(f"{name} is not found in the {self._name} registry")
        return self._module_dict[name]

# Instantiate here the needed Registries
MODELS_REGISTRY = Registry("MODELS")
DATASETS_REGISTRY = Registry("DATASETS")
# CONFIGS_REGISTRY = Registry("CONFIGS")
# add registry here

### here the example of the usage:
def main():

    Registry.show_all()
    # Class function avoiding print each registry like below:
    print(f'Registry {MODELS_REGISTRY._name} contains {MODELS_REGISTRY._module_dict}')
    print(f'Registry {DATASETS_REGISTRY._name} contains {DATASETS_REGISTRY._module_dict}')
    
    # print(dir(Registry))

    # -------- CONFIGS --------
    ## Pretend handled with hydra/omegaconf
    model_config = {"model_name": "modelA", 
                    "d_model": 256,
                    "learning_rate": 0.001
                    }
    data_config = {"data_name": "weather", 
                   "train_portion": 0.7,
                   "val_portion": 0.15
                   }

    # -------- MODEL --------
    # Init one of the models already inited
    model_class = MODELS_REGISTRY.get(model_config["model_name"])
    model = model_class(model_config)

    # -------- DATA --------
    dataset_class = DATASETS_REGISTRY.get(data_config["data_name"])
    dataset = dataset_class(data_config)

    print(f"\nUsing model {model.name} for dataset {dataset.name}")
    print("--- DAJE ROMAAA ---")

if __name__ == "__main__":

    verbose = True
    
    ## Pretend these class-inits were in the appropriate files
    @MODELS_REGISTRY.register("modelA", verbose=verbose)
    class Model_A:
        def __init__(self, config):
            self.name: str = config['model_name']
            print("\nInitializing...")
            for k,v in config.items():
                print(f'> {k:10}: {v}')

    @MODELS_REGISTRY.register("modelB", verbose=verbose)
    class Model_B:
        def __init__(self, config):
            self.name: str = config['model_name']
            print("\nInitializing...")
            for k,v in config.items():
                print(f'> {k:10}: {v}')

    @MODELS_REGISTRY.register("modelC", verbose=verbose)
    class Model_C:
        def __init__(self, config):
            pass

    @DATASETS_REGISTRY.register("weather", verbose=verbose)
    class Weather:
        def __init__(self, config):
            self.name: str = config['data_name']
            print("\nInitializing...")
            for k,v in config.items():
                print(f'> {k:10}: {v}')
    
    ## Normal pipeline steps where model and data are loaded
    main()
