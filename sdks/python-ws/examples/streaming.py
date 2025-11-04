"""
Streaming example: Generate text with streaming tokens
"""

import asyncio
from realm import RealmWebSocketClient


async def main():
    client = RealmWebSocketClient(url="ws://localhost:8080")
    
    try:
        await client.connect()
        print("✓ Connected!")

        print("\nGenerating with streaming...")
        
        # Note: Full streaming implementation pending server support
        # For now, this will yield the complete result
        async for token in client.generate_stream({
            "prompt": "Tell me a short story about a robot.",
            "max_tokens": 100,
            "temperature": 0.8,
        }):
            print(token, end="", flush=True)
        
        print("\n✓ Complete!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

