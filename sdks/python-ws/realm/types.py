"""
Type definitions for Realm WebSocket protocol
"""

from typing import Dict, Any, Optional, TypedDict, List


class GenerationOptions(TypedDict, total=False):
    """Options for text generation"""
    prompt: str
    max_tokens: int
    temperature: float
    stream: bool


class GenerationResult(TypedDict, total=False):
    """Result from text generation"""
    text: str
    tokens_generated: int
    prompt_tokens: int
    cost_usd: float
    time_ms: int


class TokenData(TypedDict):
    """Streaming token data"""
    token: str
    index: int
    is_final: bool


class PipelineInput(TypedDict):
    """Input for pipeline execution"""
    pass  # Can contain any key-value pairs


class HealthStatus(TypedDict, total=False):
    """Server health status"""
    status: str
    version: str
    uptime_seconds: int


class ParamMetadata(TypedDict):
    """Parameter metadata"""
    name: str
    param_type: str
    description: str
    required: bool
    default: Optional[Any]


class FunctionMetadata(TypedDict):
    """Function metadata"""
    name: str
    description: str
    params: List[ParamMetadata]
    returns: str
    streaming: bool


class RuntimeMetadata(TypedDict):
    """Runtime metadata"""
    version: str
    functions: List[FunctionMetadata]


class ErrorInfo(TypedDict, total=False):
    """Error information"""
    code: str
    message: str
    details: Any
    retry_after: int


class FunctionResponse(TypedDict, total=False):
    """Function response"""
    id: str
    status: str  # "streaming" | "complete" | "error" | "cancelled"
    data: Any
    error: ErrorInfo

