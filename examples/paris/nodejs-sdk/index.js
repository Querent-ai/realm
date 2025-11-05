/**
 * Paris Generation Example - Node.js WebSocket SDK
 * 
 * This example demonstrates using Realm's Node.js WebSocket SDK to generate "Paris":
 * - Question: "What is the capital of France?"
 * - Expected Answer: "Paris"
 * 
 * This requires a running Realm server (see examples/paris/server/)
 */

import { RealmWebSocketClient } from '@realm-ai/ws-client';

async function main() {
    console.log('🚀 Realm Paris Generation - Node.js SDK\n');

    // Connect to Realm server
    const client = new RealmWebSocketClient({
        url: process.env.REALM_URL || 'ws://localhost:8080',
        apiKey: process.env.REALM_API_KEY, // Optional
        model: process.env.REALM_MODEL || 'tinyllama-1.1b.Q4_K_M.gguf',
        tenantId: process.env.REALM_TENANT_ID, // Optional (auto-assigned if not provided)
    });

    try {
        console.log('📡 Connecting to Realm server...');
        await client.connect();
        console.log('✅ Connected!\n');

        // Check health
        console.log('🏥 Checking server health...');
        const health = await client.health();
        console.log(`   Status: ${health.status}\n`);

        // Generate "Paris" response
        console.log('🎯 Generating response to: "What is the capital of France?"');
        console.log('   (Expected: "Paris")\n');

        const result = await client.generate({
            prompt: 'What is the capital of France?',
            max_tokens: 20,
            temperature: 0.0, // Deterministic
            top_p: 0.9,
            top_k: 40,
        });

        console.log('✅ Generation complete!\n');
        console.log(`📝 Response: ${result.text}\n`);

        // Check for "Paris"
        if (result.text.toLowerCase().includes('paris')) {
            console.log('✅ SUCCESS: Model correctly identified Paris as the capital of France!');
        } else {
            console.log(`⚠️  Expected "Paris" in response, got: ${result.text}`);
        }

        // Show token counts
        console.log(`\n📊 Statistics:`);
        console.log(`   Input tokens: ${result.input_tokens || 'N/A'}`);
        console.log(`   Output tokens: ${result.tokens_generated || 'N/A'}`);
        console.log(`   Model: ${client.getModel()}`);
        console.log(`   Tenant ID: ${client.getTenantId()}`);

    } catch (error) {
        console.error('❌ Error:', error.message);
        if (error.message.includes('ECONNREFUSED')) {
            console.error('\n💡 Make sure the Realm server is running:');
            console.error('   cd examples/paris/server');
            console.error('   cargo run --release');
        }
        process.exit(1);
    } finally {
        await client.disconnect();
        console.log('\n👋 Disconnected');
    }
}

main().catch(console.error);

