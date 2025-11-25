#!/usr/bin/env node
/**
 * E2E Test Runner
 * 
 * Starts the server, runs all tests, then stops the server.
 */

import { spawn } from 'child_process';
import { promisify } from 'util';
import { exec } from 'child_process';
import fetch from 'node-fetch';
import { readFileSync } from 'fs';
import { join, resolve } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const execAsync = promisify(exec);

// Get absolute paths
const PROJECT_ROOT = resolve(__dirname, '..');
const SERVER_URL = process.env.REALM_SERVER_URL || 'http://localhost:3001'; // HTTP port is +1 from WebSocket
const SERVER_PORT = process.env.REALM_SERVER_PORT || '3000';
const HTTP_PORT = parseInt(SERVER_PORT) + 1;
const WASM_PATH = process.env.REALM_WASM_PATH || resolve(PROJECT_ROOT, 'crates/realm-wasm/pkg-server/realm_wasm_bg.wasm');
const MODEL_PATH = process.env.REALM_MODEL_PATH || (process.env.HOME + '/.realm/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf');

let serverProcess = null;
let serverLogs = [];

/**
 * Start the Realm server
 */
async function startServer() {
    console.log('🚀 Starting Realm server...\n');
    
    // Check if WASM file exists
    try {
        readFileSync(WASM_PATH);
    } catch (e) {
        console.error(`❌ WASM file not found: ${WASM_PATH}`);
        console.error('   Set REALM_WASM_PATH environment variable or ensure WASM is built');
        process.exit(1);
    }

    // Check if model file exists
    try {
        readFileSync(MODEL_PATH);
    } catch (e) {
        console.error(`❌ Model file not found: ${MODEL_PATH}`);
        console.error('   Set REALM_MODEL_PATH environment variable or download a model');
        process.exit(1);
    }

    // Start server
    const serverCmd = `cargo run --bin realm -- serve --wasm ${WASM_PATH} --model ${MODEL_PATH} --port ${SERVER_PORT} --http`;
    
    console.log(`   Command: ${serverCmd}\n`);
    console.log(`   WASM Path: ${WASM_PATH}`);
    console.log(`   Model Path: ${MODEL_PATH}\n`);
    
    serverProcess = spawn('cargo', ['run', '--bin', 'realm', '--', 'serve', 
        '--wasm', WASM_PATH,
        '--model', MODEL_PATH,
        '--port', SERVER_PORT,
        '--http'
    ], {
        cwd: PROJECT_ROOT,
        stdio: ['ignore', 'pipe', 'pipe'],
        shell: false
    });

    // Capture logs
    serverProcess.stdout.on('data', (data) => {
        const log = data.toString();
        serverLogs.push(log);
        process.stdout.write(`[SERVER] ${log}`);
    });

    serverProcess.stderr.on('data', (data) => {
        const log = data.toString();
        serverLogs.push(log);
        process.stderr.write(`[SERVER] ${log}`);
    });

    // Wait for server to be ready
    console.log('   Waiting for server to start...');
    let attempts = 0;
    const maxAttempts = 120; // 120 seconds max (model loading can take time)

    while (attempts < maxAttempts) {
        try {
            const response = await fetch(`${SERVER_URL}/health`, {
                method: 'GET',
                signal: AbortSignal.timeout(2000)
            });
            
            if (response.ok) {
                console.log(`   ✅ Server is ready at ${SERVER_URL}!\n`);
                return true;
            }
        } catch (e) {
            // Server not ready yet - show progress every 5 seconds
            if (attempts % 5 === 0) {
                process.stdout.write(`   Waiting... (${attempts}/${maxAttempts})\r`);
            }
        }
        
        await new Promise(resolve => setTimeout(resolve, 1000));
        attempts++;
    }

    console.error('   ❌ Server failed to start within timeout');
    return false;
}

/**
 * Stop the Realm server
 */
async function stopServer() {
    if (serverProcess) {
        console.log('\n🛑 Stopping server...');
        serverProcess.kill('SIGTERM');
        
        // Wait for process to exit
        await new Promise((resolve) => {
            serverProcess.on('exit', resolve);
            setTimeout(resolve, 5000); // Force exit after 5s
        });
        
        serverProcess = null;
        console.log('   ✅ Server stopped\n');
    }
}

/**
 * Run a test file
 */
async function runTest(testFile) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Running: ${testFile}`);
    console.log('='.repeat(60));
    
    try {
        const { execPath } = process;
        const { stdout, stderr } = await execAsync(`node ${testFile}`, {
            cwd: __dirname,
            env: { ...process.env, REALM_SERVER_URL: SERVER_URL, REALM_SERVER_PORT: SERVER_PORT }
        });
        
        if (stdout) process.stdout.write(stdout);
        if (stderr) process.stderr.write(stderr);
        
        return true;
    } catch (e) {
        console.error(`❌ Test failed: ${e.message}`);
        if (e.stdout) console.error(e.stdout);
        if (e.stderr) console.error(e.stderr);
        return false;
    }
}

/**
 * Main test runner
 */
async function main() {
    const tests = [
        'test-paris.js',
        'test-batching.js',
        'test-lora.js',
        'test-speculative.js'
    ];

    let serverStarted = false;
    
    try {
        // Start server
        serverStarted = await startServer();
        if (!serverStarted) {
            console.error('❌ Failed to start server');
            process.exit(1);
        }

        // Run tests
        const results = [];
        for (const test of tests) {
            const passed = await runTest(test);
            results.push({ test, passed });
        }

        // Print summary
        console.log('\n' + '='.repeat(60));
        console.log('📊 Test Summary');
        console.log('='.repeat(60));
        
        let passedCount = 0;
        for (const { test, passed } of results) {
            const status = passed ? '✅ PASSED' : '❌ FAILED';
            console.log(`  ${status}: ${test}`);
            if (passed) passedCount++;
        }
        
        console.log(`\n  Total: ${results.length} tests`);
        console.log(`  Passed: ${passedCount}`);
        console.log(`  Failed: ${results.length - passedCount}\n`);

        process.exit(passedCount === results.length ? 0 : 1);
    } catch (error) {
        console.error('❌ Fatal error:', error);
        process.exit(1);
    } finally {
        if (serverStarted) {
            await stopServer();
        }
    }
}

// Handle process termination
process.on('SIGINT', async () => {
    await stopServer();
    process.exit(1);
});

process.on('SIGTERM', async () => {
    await stopServer();
    process.exit(1);
});

main();

