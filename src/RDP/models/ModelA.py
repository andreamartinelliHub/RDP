from ..registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("ModelA")
class TransformerModel:
    def __init__(self, config):
        print("Initializing ModelA...")