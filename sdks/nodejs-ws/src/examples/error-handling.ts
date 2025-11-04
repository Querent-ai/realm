/**
 * Error handling example: Demonstrate error handling patterns
 */

import { RealmWebSocketClient } from "../client";

async function main() {
  const client = new RealmWebSocketClient({
    url: "ws://localhost:8080",
    apiKey: process.env.REALM_API_KEY,
  });

  try {
    await client.connect();

    // Example 1: Rate limit handling
    try {
      await client.generate({
        prompt: "Hello",
        max_tokens: 50,
      });
    } catch (error: any) {
      if (error.code === "RATE_LIMIT_EXCEEDED") {
        console.log(`Rate limited! Retry after: ${error.retryAfter} seconds`);
        // Wait and retry
        await new Promise(resolve => setTimeout(resolve, error.retryAfter * 1000));
        // Retry...
      } else {
        throw error;
      }
    }

    // Example 2: Authentication error
    try {
      const invalidClient = new RealmWebSocketClient({
        url: "ws://localhost:8080",
        apiKey: "invalid-key",
      });
      await invalidClient.connect();
    } catch (error: any) {
      if (error.code === "UNAUTHORIZED" || error.message.includes("auth")) {
        console.log("Authentication failed - check your API key");
      }
    }

    // Example 3: Function not found
    try {
      // This would fail if the function doesn't exist
      await client.generate({
        prompt: "Test",
        max_tokens: 10,
      });
    } catch (error: any) {
      if (error.code === "FUNCTION_NOT_FOUND") {
        console.log("Function not available on this server");
      } else {
        console.log("Other error:", error.message);
      }
    }

  } catch (error) {
    console.error("Unexpected error:", error);
  } finally {
    client.disconnect();
  }
}

main().catch(console.error);

