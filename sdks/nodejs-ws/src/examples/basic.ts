/**
 * Basic example: Connect to Realm server and generate text
 */

import { RealmWebSocketClient } from "../client";

async function main() {
  const client = new RealmWebSocketClient({
    url: "ws://localhost:8080",
    model: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    apiKey: process.env.REALM_API_KEY, // Optional
    tenantId: "example-tenant", // Optional
  });

  try {
    // Connect to server
    console.log("Connecting to Realm server...");
    await client.connect();
    console.log("✓ Connected!");

    // Check health
    console.log("\nChecking server health...");
    const health = await client.health();
    console.log("Health:", health);

    // Get metadata
    console.log("\nGetting runtime metadata...");
    const metadata = await client.metadata();
    console.log("Available functions:", metadata.functions.map((f: any) => f.name));

    // Generate text
    console.log("\nGenerating text...");
    const result = await client.generate({
      prompt: "What is the capital of France?",
      max_tokens: 50,
      temperature: 0.7,
    });
    console.log("Generated:", result.text);
    console.log("Tokens:", result.tokens_generated);

    // Execute pipeline (if available)
    console.log("\nExecuting pipeline...");
    try {
      const pipelineResult = await client.executePipeline("simple-chat", {
        prompt: "Hello, how are you?",
      });
      console.log("Pipeline result:", pipelineResult);
    } catch (error) {
      console.log("Pipeline not available:", (error as Error).message);
    }

  } catch (error) {
    console.error("Error:", error);
  } finally {
    client.disconnect();
  }
}

main().catch(console.error);

