from ..registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("modelA")
class ModelA:
    def __init__(self, config):
        print("Initializing ModelA...")