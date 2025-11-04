"""
Basic example: Connect to Realm server and generate text
"""

import asyncio
from realm import RealmWebSocketClient


async def main():
    client = RealmWebSocketClient(
        url="ws://localhost:8080",
        api_key=None,  # Optional: Set to your API key
        tenant_id="example-tenant",  # Optional
    )

    try:
        # Connect to server
        print("Connecting to Realm server...")
        await client.connect()
        print("✓ Connected!")

        # Check health
        print("\nChecking server health...")
        health = await client.health()
        print(f"Health: {health}")

        # Get metadata
        print("\nGetting runtime metadata...")
        metadata = await client.metadata()
        functions = [f["name"] for f in metadata.get("functions", [])]
        print(f"Available functions: {functions}")

        # Generate text
        print("\nGenerating text...")
        result = await client.generate({
            "prompt": "What is the capital of France?",
            "max_tokens": 50,
            "temperature": 0.7,
        })
        print(f"Generated: {result['text']}")
        print(f"Tokens: {result['tokens_generated']}")

        # Execute pipeline (if available)
        print("\nExecuting pipeline...")
        try:
            pipeline_result = await client.execute_pipeline("simple-chat", {
                "prompt": "Hello, how are you?",
            })
            print(f"Pipeline result: {pipeline_result}")
        except Exception as e:
            print(f"Pipeline not available: {e}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

