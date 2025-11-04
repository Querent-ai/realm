"""
Error handling example: Demonstrate error handling patterns
"""

import asyncio
from realm import RealmWebSocketClient
from realm.exceptions import RateLimitError, AuthenticationError


async def main():
    client = RealmWebSocketClient(url="ws://localhost:8080")
    
    try:
        await client.connect()

        # Example 1: Rate limit handling
        try:
            await client.generate({
                "prompt": "Hello",
                "max_tokens": 50,
            })
        except Exception as e:
            if hasattr(e, "code") and e.code == "RATE_LIMIT_EXCEEDED":
                retry_after = getattr(e, "retry_after", 60)
                print(f"Rate limited! Retry after: {retry_after} seconds")
                await asyncio.sleep(retry_after)
                # Retry...
            else:
                raise

        # Example 2: Authentication error
        try:
            invalid_client = RealmWebSocketClient(
                url="ws://localhost:8080",
                api_key="invalid-key",
            )
            await invalid_client.connect()
        except Exception as e:
            if "auth" in str(e).lower() or "unauthorized" in str(e).lower():
                print("Authentication failed - check your API key")
            await invalid_client.disconnect()

        # Example 3: Function not found
        try:
            await client.generate({
                "prompt": "Test",
                "max_tokens": 10,
            })
        except Exception as e:
            if hasattr(e, "code") and e.code == "FUNCTION_NOT_FOUND":
                print("Function not available on this server")
            else:
                print(f"Other error: {e}")

    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

