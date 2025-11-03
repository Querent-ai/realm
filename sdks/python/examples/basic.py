"""
Basic example using Realm.ai Python SDK (HTTP client mode)
"""

from realm import RealmClient

def main():
    # Initialize client
    client = RealmClient(
        base_url="http://localhost:8080",
        api_key=None,  # Optional
    )

    try:
        # Simple completion
        response = client.completions(
            prompt="What is the capital of France?",
            max_tokens=50,
            temperature=0.7,
        )

        print("Response:", response.get("text", ""))
        print("Model:", response.get("model", "unknown"))

        # List available models
        models = client.models()
        print("\nAvailable models:")
        for model in models.get("models", []):
            print(f"  - {model.get('id')}: {model.get('name')}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    main()

