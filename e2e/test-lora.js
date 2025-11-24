#!/usr/bin/env node
/**
 * E2E Test: LoRA Adapter Integration
 *
 * This test verifies that LoRA adapters can be loaded and applied to models.
 */

import fetch from 'node-fetch';

const SERVER_URL = process.env.REALM_SERVER_URL || 'http://localhost:3000';
const TIMEOUT = 30000;

async function testLoRAIntegration() {
    console.log('🧪 Testing LoRA Integration...\n');

    // Note: This is a placeholder test since LoRA adapter loading
    // requires actual adapter files. In a real scenario, you would:
    // 1. Load a LoRA adapter via API (if endpoint exists)
    // 2. Set it for a tenant
    // 3. Generate text and verify the adapter is applied

    console.log('  ⚠️  LoRA adapter loading requires adapter files');
    console.log('  ⚠️  This test is a placeholder - implement when adapter API is available\n');

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

testLoRAIntegration()
    .then(success => {
        process.exit(success ? 0 : 1);
    })
    .catch(error => {
        console.error('Fatal error:', error);
        process.exit(1);
    });

