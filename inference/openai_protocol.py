from typing import List, Literal, Optional
from pydantic import BaseModel


class ChatCompletionMessage(BaseModel):
    role: Literal["user", "system", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatCompletionMessage]
    sampling_params: Optional[dict] = None
