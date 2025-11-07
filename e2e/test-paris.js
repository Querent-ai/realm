#!/usr/bin/env node
/**
 * E2E Test: Paris Generation Verification
 * 
 * This test verifies that the server correctly responds to "capital of France" 
 * prompts with "Paris" using the WASM runtime.
 */

import fetch from 'node-fetch';

const SERVER_URL = process.env.REALM_SERVER_URL || 'http://localhost:3000';
const TIMEOUT = 30000; // 30 seconds

async function testParisGeneration() {
    console.log('🧪 Testing Paris Generation...\n');
    
    const testCases = [
        {
            name: 'Direct question',
            prompt: 'What is the capital of France?',
            expected: 'Paris'
        },
        {
            name: 'Capital of France',
            prompt: 'capital of France',
            expected: 'Paris'
        },
        {
            name: 'France capital',
            prompt: 'France capital',
            expected: 'Paris'
        }
    ];

    let passed = 0;
    let failed = 0;

    for (const testCase of testCases) {
        try {
            console.log(`  Testing: ${testCase.name} (prompt: "${testCase.prompt}")`);
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), TIMEOUT);
            
            const response = await fetch(`${SERVER_URL}/v1/completions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: testCase.prompt,
                    max_tokens: 50,
                    stream: false
                }),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            const text = data.choices?.[0]?.text || data.text || '';
            
            console.log(`    Response: "${text.trim()}"`);
            
            if (text.toLowerCase().includes(testCase.expected.toLowerCase())) {
                console.log(`    ✅ PASSED\n`);
                passed++;
            } else {
                console.log(`    ❌ FAILED: Expected to contain "${testCase.expected}"\n`);
                failed++;
            }
        } catch (error) {
            console.log(`    ❌ ERROR: ${error.message}\n`);
            failed++;
        }
    }

    // Test streaming
    try {
        console.log('  Testing: Streaming response');
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), TIMEOUT);
        
        const response = await fetch(`${SERVER_URL}/v1/completions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: 'What is the capital of France?',
                max_tokens: 50,
                stream: true
            }),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const reader = response.body;
        let chunks = [];
        let decoder = new TextDecoder();
        
        for await (const chunk of reader) {
            const text = decoder.decode(chunk);
            const lines = text.split('\n').filter(line => line.trim() && line.startsWith('data:'));
            
            for (const line of lines) {
                const data = line.slice(5).trim();
                if (data === '[DONE]') continue;
                
                try {
                    const json = JSON.parse(data);
                    if (json.choices?.[0]?.delta?.content) {
                        chunks.push(json.choices[0].delta.content);
                    }
                } catch (e) {
                    // Skip invalid JSON
                }
            }
        }

        const streamedText = chunks.join('');
        console.log(`    Streamed response: "${streamedText.trim()}"`);
        
        if (streamedText.toLowerCase().includes('paris')) {
            console.log(`    ✅ PASSED\n`);
            passed++;
        } else {
            console.log(`    ❌ FAILED: Expected to contain "Paris"\n`);
            failed++;
        }
    } catch (error) {
        console.log(`    ❌ ERROR: ${error.message}\n`);
        failed++;
    }

    console.log('\n📊 Results:');
    console.log(`  ✅ Passed: ${passed}`);
    console.log(`  ❌ Failed: ${failed}`);
    console.log(`  Total: ${passed + failed}\n`);

    return failed === 0;
}

// Run tests
testParisGeneration()
    .then(success => {
        process.exit(success ? 0 : 1);
    })
    .catch(error => {
        console.error('Fatal error:', error);
        process.exit(1);
    });

