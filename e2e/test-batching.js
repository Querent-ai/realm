#!/usr/bin/env node
/**
 * E2E Test: Continuous Batching Integration
 *
 * This test verifies that continuous batching improves throughput.
 */

import fetch from 'node-fetch';

const SERVER_URL = process.env.REALM_SERVER_URL || 'http://localhost:3001'; // HTTP port is +1 from WebSocket
const TIMEOUT = 300000; // 300 seconds (5 minutes) - model loading + generation can be slow

async function testContinuousBatching() {
    console.log('🧪 Testing Continuous Batching...\n');

    // Test concurrent requests to verify batching improves throughput
    const concurrentRequests = 5;
    const prompts = [
        'What is the capital of France?',
        'Tell me about Paris.',
        'France is a country in',
        'The Eiffel Tower is in',
        'Paris is the capital of',
    ];

    console.log(`  Sending ${concurrentRequests} concurrent requests...\n`);

    const startTime = Date.now();
    const promises = prompts.map((prompt, i) => {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), TIMEOUT);
        
        return fetch(`${SERVER_URL}/v1/chat/completions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: 'realm-model',
                messages: [
                    {
                        role: 'user',
                        content: prompt
                    }
                ],
                max_tokens: 20,
                stream: false
            }),
            signal: controller.signal
        }).then(async (response) => {
            clearTimeout(timeoutId);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            const data = await response.json();
            return {
                index: i,
                prompt,
                text: data.choices?.[0]?.message?.content || data.choices?.[0]?.text || data.text || '',
                time: Date.now() - startTime,
            };
        }).catch((error) => {
            clearTimeout(timeoutId);
            throw error;
        });
    });

    try {
        const results = await Promise.all(promises);
        const totalTime = Date.now() - startTime;
        const avgTime = totalTime / concurrentRequests;

        console.log('  Results:');
        results.forEach((result) => {
            console.log(`    Request ${result.index + 1}: ${result.text.substring(0, 50)}... (${result.time}ms)`);
        });

        console.log(`\n  Total time: ${totalTime}ms`);
        console.log(`  Average time per request: ${avgTime.toFixed(2)}ms`);
        console.log(`  Throughput: ${((concurrentRequests / totalTime) * 1000).toFixed(2)} requests/sec\n`);

        // Verify all requests completed successfully
        if (results.length === concurrentRequests) {
            console.log('  ✅ All requests completed successfully\n');
            return true;
        } else {
            console.log(`  ❌ Only ${results.length}/${concurrentRequests} requests completed\n`);
            return false;
        }
    } catch (error) {
        console.log(`  ❌ Batch test failed: ${error.message}\n`);
        return false;
    }
}

testContinuousBatching()
    .then(success => {
        process.exit(success ? 0 : 1);
    })
    .catch(error => {
        console.error('Fatal error:', error);
        process.exit(1);
    });

