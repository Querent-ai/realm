"""
HTTP client for Realm.ai API (Server mode)

For local/WASM mode, see future implementation.
Currently implements HTTP client for connecting to realm-runtime server.
"""

from typing import Optional, Dict, Any, Generator, AsyncGenerator
import httpx
from .exceptions import RealmError, TimeoutError, RateLimitError


class RealmClient:
    """
    HTTP client for Realm.ai API server mode.
    
    Note: For local/WASM mode (Python bindings), see future implementation.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize Realm HTTP client.

        Args:
            base_url: Base URL for Realm API server
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            headers: Custom headers
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        default_headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            default_headers["Authorization"] = f"Bearer {self.api_key}"

        if headers:
            default_headers.update(headers)

        self.headers = default_headers

        # Create httpx client
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self.headers,
            timeout=self.timeout,
        )

        # Async client (lazy initialization)
        self._async_client = None

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async client"""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=self.timeout,
            )
        return self._async_client

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        try:
            response = self._client.request(
                method=method,
                url=endpoint,
                json=data,
            )

            if not response.is_success:
                error_body = response.json() if response.content else {}

                if response.status_code == 429:
                    raise RateLimitError(
                        error_body.get("message", "Rate limit exceeded")
                    )

                raise RealmError(
                    error_body.get("message", f"HTTP {response.status_code}"),
                    status_code=response.status_code,
                    code=error_body.get("code"),
                    response=error_body,
                )

            return response.json()

        except httpx.HTTPError as e:
            if retry_count < self.max_retries:
                import time

                time.sleep(2 ** retry_count)
                return self._request(method, endpoint, data, retry_count + 1)

            raise TimeoutError(f"Request failed: {str(e)}")

    def completions(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Create a completion"""
        data = {"prompt": prompt}

        if model:
            data["model"] = model
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if temperature is not None:
            data["temperature"] = temperature
        if top_k is not None:
            data["top_k"] = top_k
        if top_p is not None:
            data["top_p"] = top_p
        if stop is not None:
            data["stop"] = stop

        return self._request("POST", "/v1/completions", data)

    def models(self) -> Dict[str, Any]:
        """Get list of available models"""
        return self._request("GET", "/v1/models")

    def health(self) -> Dict[str, str]:
        """Health check"""
        return self._request("GET", "/health")

    def close(self):
        """Close the client"""
        self._client.close()
        if self._async_client:
            # Note: AsyncClient.close() is async, but this is sync method
            # In real usage, use async context manager
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_realm(
    base_url: str = "http://localhost:8080",
    api_key: Optional[str] = None,
    **kwargs,
) -> RealmClient:
    """Create a new Realm HTTP client"""
    return RealmClient(base_url=base_url, api_key=api_key, **kwargs)

