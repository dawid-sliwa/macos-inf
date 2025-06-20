from typing import Union
import warnings
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoConfig,
)


def get_tokenizer(
    tokenizer_name: str,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return tokenizer
    except Exception as exc:
        warnings.warn("Some error during error loading")
        raise exc


def get_config(model: str) -> AutoConfig:
    print("trying to download config from hugging face")
    config = AutoConfig.from_pretrained(model)
    print("downloaded model from hugging face")

    return config
