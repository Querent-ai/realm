"""
Paris Generation Example - Python WebSocket SDK

This example demonstrates using Realm's Python WebSocket SDK to generate "Paris":
- Question: "What is the capital of France?"
- Expected Answer: "Paris"

This requires a running Realm server (see examples/paris/server/)
"""

import asyncio
import os
from realm import RealmWebSocketClient


async def main():
    print("🚀 Realm Paris Generation - Python SDK\n")

    # Connect to Realm server
    client = RealmWebSocketClient(
        url=os.getenv("REALM_URL", "ws://localhost:8080"),
        api_key=os.getenv("REALM_API_KEY"),  # Optional
        model=os.getenv("REALM_MODEL", "tinyllama-1.1b.Q4_K_M.gguf"),
        tenant_id=os.getenv("REALM_TENANT_ID"),  # Optional (auto-assigned)
    )

    try:
        print("📡 Connecting to Realm server...")
        await client.connect()
        print("✅ Connected!\n")

        # Check health
        print("🏥 Checking server health...")
        health = await client.health()
        print(f"   Status: {health.get('status', 'unknown')}\n")

        # Generate "Paris" response
        print('🎯 Generating response to: "What is the capital of France?"')
        print('   (Expected: "Paris")\n')

        result = await client.generate({
            "prompt": "What is the capital of France?",
            "max_tokens": 20,
            "temperature": 0.0,  # Deterministic
            "top_p": 0.9,
            "top_k": 40,
        })

        print("✅ Generation complete!\n")
        print(f"📝 Response: {result.get('text', 'N/A')}\n")

        # Check for "Paris"
        response_text = result.get("text", "").lower()
        if "paris" in response_text:
            print("✅ SUCCESS: Model correctly identified Paris as the capital of France!")
        else:
            print(f"⚠️  Expected 'Paris' in response, got: {result.get('text', 'N/A')}")

        # Show token counts
        print(f"\n📊 Statistics:")
        print(f"   Input tokens: {result.get('input_tokens', 'N/A')}")
        print(f"   Output tokens: {result.get('tokens_generated', 'N/A')}")
        print(f"   Model: {client.get_model()}")
        print(f"   Tenant ID: {client.get_tenant_id()}")

    except Exception as e:
        print(f"❌ Error: {e}")
        if "Connection refused" in str(e) or "ECONNREFUSED" in str(e):
            print("\n💡 Make sure the Realm server is running:")
            print("   cd examples/paris/server")
            print("   cargo run --release")
        import sys
        sys.exit(1)
    finally:
        await client.disconnect()
        print("\n👋 Disconnected")


if __name__ == "__main__":
    asyncio.run(main())

