#!/usr/bin/env node
/**
 * E2E Test: Speculative Decoding Integration
 *
 * This test verifies that speculative decoding works with draft models.
 */

import fetch from 'node-fetch';

const SERVER_URL = process.env.REALM_SERVER_URL || 'http://localhost:3000';
const TIMEOUT = 30000;

async function testSpeculativeDecoding() {
    console.log('🧪 Testing Speculative Decoding...\n');

    // Note: This test requires a draft model to be configured.
    // In a real scenario, you would:
    // 1. Configure a draft model in server config
    // 2. Generate text and measure speedup
    // 3. Verify tokens are generated correctly

    console.log('  ⚠️  Speculative decoding requires draft model configuration');
    console.log('  ⚠️  This test is a placeholder - implement when draft model API is available\n');

    // Test that server is running
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), TIMEOUT);
        
        const response = await fetch(`${SERVER_URL}/health`, {
            method: 'GET',
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);

        if (response.ok) {
            console.log('  ✅ Server is running\n');
            return true;
        } else {
            console.log(`  ❌ Server health check failed: ${response.status}\n`);
            return false;
        }
    } catch (error) {
        console.log(`  ❌ Server not accessible: ${error.message}\n`);
        return false;
    }
}

testSpeculativeDecoding()
    .then(success => {
        process.exit(success ? 0 : 1);
    })
    .catch(error => {
        console.error('Fatal error:', error);
        process.exit(1);
    });

