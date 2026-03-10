from ..registry import MODELS_REGISTRY

@MODELS_REGISTRY.register("modelA")
class ModelA:
    def __init__(self, config):
        print("Initializing ModelA...")