/**
 * Streaming example: Generate text with streaming tokens
 */

import { RealmWebSocketClient } from "../client";

async function main() {
  const client = new RealmWebSocketClient({
    url: "ws://localhost:8080",
    model: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    apiKey: process.env.REALM_API_KEY,
  });

  try {
    await client.connect();
    console.log("✓ Connected!");

    console.log("\nGenerating with streaming...");
    
    for await (const token of client.generateStream({
      prompt: "Tell me a short story about a robot.",
      max_tokens: 100,
      temperature: 0.8,
    })) {
      process.stdout.write(token);
    }
    
    console.log("\n✓ Complete!");

  } catch (error) {
    console.error("Error:", error);
  } finally {
    client.disconnect();
  }
}

main().catch(console.error);

