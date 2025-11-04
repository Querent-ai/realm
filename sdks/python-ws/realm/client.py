"""
Realm WebSocket Client

Connects to Realm.ai WebSocket server and provides type-safe function dispatch.
"""

import asyncio
import json
import uuid
from typing import Optional, Dict, Any, Callable, AsyncGenerator
import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI
from .types import (
    GenerationOptions,
    GenerationResult,
    PipelineInput,
    FunctionResponse,
    ErrorInfo,
)


class RealmWebSocketClient:
    """WebSocket client for Realm.ai server"""

    def __init__(
        self,
        url: str = "ws://localhost:8080",
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        reconnect: bool = True,
        reconnect_interval: float = 5.0,
        timeout: float = 30.0,
    ):
        """
        Initialize Realm WebSocket client.

        Args:
            url: WebSocket URL (default: "ws://localhost:8080")
            api_key: API key for authentication
            tenant_id: Tenant ID for multi-tenancy
            reconnect: Enable automatic reconnection (default: True)
            reconnect_interval: Reconnection interval in seconds (default: 5.0)
            timeout: Request timeout in seconds (default: 30.0)
        """
        self.url = url
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.reconnect = reconnect
        self.reconnect_interval = reconnect_interval
        self.timeout = timeout
        
        self._ws: Optional[websockets.WebSocketServerProtocol] = None
        self._is_authenticated = False
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._streaming_callbacks: Dict[str, Callable] = {}
        self._event_callbacks: Dict[str, list] = {}
        self._error_callbacks: list = []
        self._reconnect_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Connect to the WebSocket server"""
        if self._ws and not self._ws.closed:
            return

        try:
            self._ws = await websockets.connect(self.url)
            self._is_authenticated = False

            # Start message handler
            asyncio.create_task(self._message_handler())

            # Authenticate if API key provided
            if self.api_key:
                await self._authenticate(self.api_key)
            
            self._is_authenticated = True
            self._emit("connected")

        except Exception as e:
            self._emit("error", e)
            for callback in self._error_callbacks:
                callback(e)
            raise

    async def _authenticate(self, api_key: str) -> None:
        """Authenticate with API key"""
        if not self._ws or self._ws.closed:
            raise ConnectionError("WebSocket not connected")

        # Create authentication request
        auth_request = {
            "type": "auth",
            "api_key": api_key,
        }

        await self._ws.send(json.dumps(auth_request))

        # Wait for authentication response (simplified - in production, use proper message handling)
        await asyncio.sleep(0.5)  # Give server time to respond

    async def _message_handler(self) -> None:
        """Handle incoming WebSocket messages"""
        if not self._ws:
            return

        try:
            async for message in self._ws:
                await self._handle_message(message)
        except ConnectionClosed:
            self._is_authenticated = False
            self._emit("disconnected")

            # Cancel pending requests
            for request_id, future in self._pending_requests.items():
                if not future.done():
                    future.set_exception(ConnectionError("Connection closed"))
            self._pending_requests.clear()

            # Attempt reconnection if enabled
            if self.reconnect and not self._reconnect_task:
                self._reconnect_task = asyncio.create_task(self._reconnect())
        except Exception as e:
            self._emit("error", e)
            for callback in self._error_callbacks:
                callback(e)

    async def _handle_message(self, text: str) -> None:
        """Process incoming message"""
        try:
            message = json.loads(text)

            # Handle authentication responses
            if message.get("type") in ("auth_success", "auth_failed"):
                self._emit("message", message)
                return

            # Handle function responses
            request_id = message.get("id")
            if request_id and request_id in self._pending_requests:
                future = self._pending_requests[request_id]

                if message.get("status") == "error":
                    error_info: ErrorInfo = message.get("error", {})
                    error = Exception(error_info.get("message", "Function call failed"))
                    error.code = error_info.get("code")  # type: ignore
                    error.retry_after = error_info.get("retry_after")  # type: ignore
                    future.set_exception(error)
                    del self._pending_requests[request_id]
                    
                    if request_id in self._streaming_callbacks:
                        del self._streaming_callbacks[request_id]

                elif message.get("status") == "complete":
                    future.set_result(message)
                    del self._pending_requests[request_id]
                    
                    if request_id in self._streaming_callbacks:
                        del self._streaming_callbacks[request_id]

                elif message.get("status") == "streaming":
                    # Handle streaming response
                    callback = self._streaming_callbacks.get(request_id)
                    if callback:
                        callback(message.get("data"))

                elif message.get("status") == "cancelled":
                    future.set_exception(Exception("Function call was cancelled"))
                    del self._pending_requests[request_id]
                    
                    if request_id in self._streaming_callbacks:
                        del self._streaming_callbacks[request_id]

        except json.JSONDecodeError as e:
            print(f"Failed to parse message: {e}")

    async def _call_function(
        self,
        function_name: str,
        params: Dict[str, Any],
        stream_callback: Optional[Callable] = None,
    ) -> FunctionResponse:
        """Send a function call and wait for response"""
        if not self._ws or self._ws.closed:
            raise ConnectionError("WebSocket not connected. Call connect() first.")

        if not self._is_authenticated:
            raise ConnectionError("Not authenticated. Wait for connect() to complete.")

        call = {
            "id": str(uuid.uuid4()),
            "function": function_name,
            "params": params,
        }

        if self.tenant_id:
            call["tenant_id"] = self.tenant_id

        future = asyncio.Future()
        self._pending_requests[call["id"]] = future

        if stream_callback:
            self._streaming_callbacks[call["id"]] = stream_callback

        try:
            await self._ws.send(json.dumps(call))

            # Wait for response with timeout
            response = await asyncio.wait_for(future, timeout=self.timeout)
            return response

        except asyncio.TimeoutError:
            del self._pending_requests[call["id"]]
            if stream_callback:
                del self._streaming_callbacks[call["id"]]
            raise TimeoutError("Function call timeout")

    async def generate(self, options: GenerationOptions) -> GenerationResult:
        """
        Generate text from a prompt.

        Args:
            options: Generation options (prompt, max_tokens, temperature, etc.)

        Returns:
            Generation result with text and metadata
        """
        response = await self._call_function("generate", {
            "prompt": options["prompt"],
            "max_tokens": options.get("max_tokens", 100),
            "temperature": options.get("temperature", 0.7),
            "stream": options.get("stream", False),
        })

        if response.get("status") == "error":
            error_info = response.get("error", {})
            raise Exception(error_info.get("message", "Generation failed"))

        return response["data"]

    async def generate_stream(
        self, options: GenerationOptions
    ) -> AsyncGenerator[str, None]:
        """
        Generate text with streaming.

        Args:
            options: Generation options (must have stream=True)

        Yields:
            Generated tokens as they arrive

        Note: Streaming requires proper handling of streaming responses.
        For now, this returns the full result. Full streaming support
        will be implemented when server streaming is fully functional.
        """
        stream_options = {**options, "stream": True}
        
        # For now, get full result and yield it
        # TODO: Implement proper streaming when server supports it
        result = await self.generate(stream_options)
        yield result["text"]

    async def execute_pipeline(
        self, pipeline_name: str, input_data: PipelineInput
    ) -> Any:
        """
        Execute a multi-model pipeline.

        Args:
            pipeline_name: Name of the pipeline to execute
            input_data: Input data for the pipeline

        Returns:
            Pipeline execution result
        """
        response = await self._call_function("pipeline", {
            "pipeline": pipeline_name,
            "input": input_data,
        })

        if response.get("status") == "error":
            error_info = response.get("error", {})
            raise Exception(error_info.get("message", "Pipeline execution failed"))

        return response["data"]

    async def health(self) -> Dict[str, Any]:
        """Check server health status"""
        response = await self._call_function("health", {})

        if response.get("status") == "error":
            error_info = response.get("error", {})
            raise Exception(error_info.get("message", "Health check failed"))

        return response["data"]

    async def metadata(self) -> Dict[str, Any]:
        """Get runtime metadata (available functions, parameters, etc.)"""
        response = await self._call_function("metadata", {})

        if response.get("status") == "error":
            error_info = response.get("error", {})
            raise Exception(error_info.get("message", "Metadata request failed"))

        return response["data"]

    def on(self, event: str, callback: Callable) -> None:
        """Register event callback"""
        if event not in self._event_callbacks:
            self._event_callbacks[event] = []
        self._event_callbacks[event].append(callback)

    def off(self, event: str, callback: Callable) -> None:
        """Remove event callback"""
        if event in self._event_callbacks:
            try:
                self._event_callbacks[event].remove(callback)
            except ValueError:
                pass

    def _emit(self, event: str, data: Any = None) -> None:
        """Emit event to callbacks"""
        if event in self._event_callbacks:
            for callback in self._event_callbacks[event]:
                callback(data)

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register error callback"""
        self._error_callbacks.append(callback)

    async def disconnect(self) -> None:
        """Disconnect from server"""
        self.reconnect = False

        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        if self._ws and not self._ws.closed:
            await self._ws.close()

        self._pending_requests.clear()
        self._streaming_callbacks.clear()
        self._is_authenticated = False

    def is_connected(self) -> bool:
        """Check if connected to server"""
        return self._ws is not None and not self._ws.closed and self._is_authenticated

    async def _reconnect(self) -> None:
        """Attempt to reconnect"""
        await asyncio.sleep(self.reconnect_interval)
        try:
            await self.connect()
        except Exception:
            pass  # Will retry on next disconnect
        finally:
            self._reconnect_task = None

