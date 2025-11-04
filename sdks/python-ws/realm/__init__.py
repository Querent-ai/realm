"""
Realm WebSocket Client SDK

Official WebSocket client for Realm.ai multi-tenant LLM inference server.
"""

from .client import RealmWebSocketClient
from .types import (
    GenerationOptions,
    GenerationResult,
    PipelineInput,
    HealthStatus,
    RuntimeMetadata,
)

__version__ = "0.1.0"
__all__ = [
    "RealmWebSocketClient",
    "GenerationOptions",
    "GenerationResult",
    "PipelineInput",
    "HealthStatus",
    "RuntimeMetadata",
]

