from RDP.registry import DATASETS_REGISTRY

@DATASETS_REGISTRY.register("weather")
class WeatherDataset:
    def __init__(self, path):
        self.path = path
    
    def load(self):
        # Implementation to return pd.DataFrame
        pass