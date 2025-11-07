"""
Tests for streaming functionality

Note: These are unit tests that verify the streaming implementation exists.
Full integration tests require a running server and are in examples/paris/.
"""

import pytest
from realm import RealmWebSocketClient


@pytest.mark.asyncio
async def test_generate_stream_method_exists():
    """Test that generate_stream method exists and has correct signature"""
    client = RealmWebSocketClient(url="ws://localhost:8080", model="test-model")
    
    # Verify method exists
    assert hasattr(client, 'generate_stream'), "generate_stream method should exist"
    
    # Verify it's callable
    assert callable(client.generate_stream), "generate_stream should be callable"
    
    # Verify signature (should accept GenerationOptions)
    import inspect
    sig = inspect.signature(client.generate_stream)
    params = list(sig.parameters.keys())
    assert 'options' in params, "generate_stream should accept 'options' parameter"
    
    print("✅ generate_stream method exists with correct signature")


@pytest.mark.asyncio
async def test_generate_stream_requires_connection():
    """Test that generate_stream requires connection"""
    client = RealmWebSocketClient(url="ws://localhost:8080", model="test-model")
    
    # Should raise error if not connected
    with pytest.raises(ConnectionError):
        async for token in client.generate_stream({
            "prompt": "Hello",
        }):
            pass
    
    print("✅ generate_stream correctly requires connection")


def test_streaming_imports():
    """Test that streaming-related types are importable"""
    from realm.types import TokenData, GenerationOptions
    
    # Verify types exist
    assert TokenData is not None
    assert GenerationOptions is not None
    
    print("✅ Streaming types are importable")

