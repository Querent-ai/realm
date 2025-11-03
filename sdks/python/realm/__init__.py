"""
Realm.ai Python SDK

Official Python SDK for Realm multi-tenant LLM inference runtime.

Currently implements HTTP client for server mode.
Local/WASM mode will be added in future (PyO3 or wasmer-python).
"""

from .client import RealmClient, create_realm
from .exceptions import RealmError, TimeoutError, RateLimitError

__version__ = "0.1.0"

__all__ = [
    "RealmClient",
    "create_realm",
    "RealmError",
    "TimeoutError",
    "RateLimitError",
]

# Convenience alias
Realm = RealmClient
