from ..registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("chronos2")
class Chronos2:
    def __init__(self, config):
        print("Initializing Chronos...")