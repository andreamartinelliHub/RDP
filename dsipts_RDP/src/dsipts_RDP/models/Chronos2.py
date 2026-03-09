from ..registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("chronos")
class ChronosModel:
    def __init__(self, config):
        print("Initializing Chronos...")