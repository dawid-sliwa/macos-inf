from typing import Union
from inference.config.model_config import ModelConfig
from inference.model_loader.model_loader import ModelLoader
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from inference.openai_protocol import ChatCompletionRequest


class AsyncLLM:
    def __init__(self, model_config: ModelConfig):
        loader = ModelLoader()

        self.loader = loader.load_model(config=model_config)
        self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = (
            AutoTokenizer.from_pretrained(model_config.model_path)
        )

    async def create_chat_completion(self, request: ChatCompletionRequest):
        request_prompt = self.tokenizer.apply_chat_template(
            conversation=request.messages
        )

        print(request_prompt)
