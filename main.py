class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register(self, name=None):
        """Decorator to register a class/function."""
        def wrapper(module):
            module_name = name if name is not None else module.__name__
            if module_name in self._module_dict:
                raise KeyError(f"{module_name} is already registered in {self._name}")
            self._module_dict[module_name] = module
            return module
        return wrapper

    def get(self, name):
        """Fetch a class by its registered string name."""
        if name not in self._module_dict:
            raise KeyError(f"{name} is not found in the {self._name} registry")
        return self._module_dict[name]

# Initialize your registries
MODEL_REGISTRY = Registry("MODELS")
DATASET_REGISTRY = Registry("DATASETS")

@MODEL_REGISTRY.register("chronos")
class ChronosModel:
    def __init__(self, config):
        print("Initializing Chronos...")

@MODEL_REGISTRY.register("modelA")
class TransformerModel:
    def __init__(self, config):
        print("Initializing modelA...")


def main():
    print("Hello from rdp!")

    config = {"model_type": "chronos", "learning_rate": 0.001}
    # This is the "Magic" line
    model_class = MODEL_REGISTRY.get(config["model_type"])
    model = model_class(config)
    
    
    config = {"model_type": "modelA", "learning_rate": 0.001}
    # This is the "Magic" line
    model_class = MODEL_REGISTRY.get(config["model_type"])
    model = model_class(config)

    print(MODEL_REGISTRY._module_dict)
    print(MODEL_REGISTRY._name)
    print(dir(MODEL_REGISTRY))


if __name__ == "__main__":
    main()
