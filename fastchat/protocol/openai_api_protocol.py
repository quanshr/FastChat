from typing import Literal, Optional, List, Dict, Any, Union

import time

import shortuuid
from pydantic import BaseModel, Field, Extra


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int


class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{shortuuid.random()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = True
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: str = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "fastchat"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = []


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class LogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)


class TokenTopLogProb(BaseModel):
    """OpenAI-style chat completion top-logprob information of a single token.

    See <https://platform.openai.com/docs/api-reference/chat/object>.
    """
    token: str
    logprob: Optional[float] = -9999.0
    bytes: Optional[List[int]] = None


class TokenLogProb(BaseModel):
    """OpenAI-style chat completion logprob information of a single token.

    See <https://platform.openai.com/docs/api-reference/chat/object>.
    """
    token: str
    logprob: Optional[float] = -9999.0
    bytes: Optional[List[int]] = None
    top_logprobs: List[TokenTopLogProb] = Field(default_factory=list)


class ChatCompletionLogProbs(BaseModel):
    """OpanAI-style chat completion logprobs.

    See <https://platform.openai.com/docs/api-reference/chat/object>.

    This is different from old style `LogProbs`,
    see <https://github.com/vllm-project/vllm/issues/3179> for more details.
    """
    content: List[TokenLogProb] = Field(default_factory=list)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: Union[str, List[Dict[str, Union[
        str,
        List[Dict[str, Any]],               # [NOTE]: Qwen-VL (+ OpenAI API)
    ]]]]
    # temperature: Optional[float] = 0.7
    temperature: Optional[float] = 1.0
    # top_p: Optional[float] = 1.0
    top_p: Optional[float] = 0.8
    min_p: Optional[float] = 0.0
    n: Optional[int] = 1
    # max_tokens: Optional[int] = None
    max_tokens: Optional[int] = 2048
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None

    # Additional parameters supported by vLLM
    top_k: Optional[int] = -1
    repetition_penalty: Optional[float] = 1.0
    best_of: Optional[int] = None
    use_beam_search: Optional[bool] = None

    echo: Optional[bool] = None
    logprobs: Optional[bool] = None
    prompt_logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = 5
    conv_template: Optional[str] = None     # [NOTE]: Per-request conv template, override default value.


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None
    logprobs: Optional[ChatCompletionLogProbs] = None
    prompt_logprobs: Optional[ChatCompletionLogProbs] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None
    logprobs: Optional[ChatCompletionLogProbs] = None
    prompt_logprobs: Optional[ChatCompletionLogProbs] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]


class TokenCheckRequestItem(BaseModel):
    model: str
    prompt: str
    max_tokens: int


class TokenCheckRequest(BaseModel):
    prompts: List[TokenCheckRequestItem]


class TokenCheckResponseItem(BaseModel):
    fits: bool
    tokenCount: int
    contextLength: int


class TokenCheckResponse(BaseModel):
    prompts: List[TokenCheckResponseItem]


class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    engine: Optional[str] = None
    input: Union[str, List[Any]]
    user: Optional[str] = None
    encoding_format: Optional[str] = None


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: UsageInfo


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[Any]]
    suffix: Optional[str] = None
    temperature: Optional[float] = 0.7
    n: Optional[int] = 1
    max_tokens: Optional[int] = 16
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0
    min_p: Optional[float] = 0.0
    top_k: Optional[int] = -1
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    use_beam_search: Optional[bool] = False
    best_of: Optional[int] = None
    # Additional parameters supported by vLLM
    repetition_penalty: Optional[float] = 1.0


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
