/**
 * Streaming example: Generate text with streaming tokens
 */

import { RealmWebSocketClient } from "../client";

async function main() {
  const client = new RealmWebSocketClient({
    url: "ws://localhost:8080",
    apiKey: process.env.REALM_API_KEY,
  });

  try {
    await client.connect();
    console.log("✓ Connected!");

    console.log("\nGenerating with streaming...");
    
    // Note: Full streaming implementation pending server support
    // For now, this will yield the complete result
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

