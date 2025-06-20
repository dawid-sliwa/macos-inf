


from inference.config.model_config import ModelConfig
from inference.model_loader.model_loader import ModelLoader


class AsyncLLM:
    def __init__(self, model_config: ModelConfig):
        
        loader = ModelLoader()

        loader.load_model(config=model_config)