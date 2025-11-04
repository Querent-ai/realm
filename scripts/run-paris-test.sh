#!/bin/bash
# Simplified Paris Test - Works with simulated server responses

set -e

echo "🧪 Paris Generation Test (Simplified)"
echo "====================================="
echo ""

# Check if server binary exists
if [ ! -f "target/release/realm" ]; then
    echo "📦 Building Realm server..."
    cargo build --release --bin realm
fi

# Check if model exists (optional for simulated responses)
MODEL_FILE=""
if [ -f "models/tinyllama-1.1b.Q4_K_M.gguf" ]; then
    MODEL_FILE="models/tinyllama-1.1b.Q4_K_M.gguf"
elif [ -f "models/llama-2-7b-chat-q4_k_m.gguf" ]; then
    MODEL_FILE="models/llama-2-7b-chat-q4_k_m.gguf"
else
    echo "⚠️  No model file found - server will use simulated responses"
    echo "   (This is fine for testing the SDK connection)"
fi

# Build SDK
echo "📦 Building Node.js SDK..."
cd sdks/nodejs-ws
npm install --silent > /dev/null 2>&1
npm run build > /dev/null 2>&1
cd ../..

# Start server
echo "🚀 Starting Realm server..."
PORT=8080

# Use WASM file
WASM_FILE="wasm-pkg/realm_wasm_bg.wasm"

# Check what we have
if [ -f "$WASM_FILE" ] && [ -f "$MODEL_FILE" ]; then
    echo "✓ Starting with WASM and model"
    ./target/release/realm serve \
        --wasm "$WASM_FILE" \
        --model "$MODEL_FILE" \
        --host 127.0.0.1 \
        --port $PORT \
        > /tmp/realm-server.log 2>&1 &
else
    echo "⚠️  Starting example server (no WASM/model required)"
    # Use the example server which doesn't require WASM/model
    cd examples/websocket-server
    cargo run --release \
        > /tmp/realm-server.log 2>&1 &
    cd ../..
fi

SERVER_PID=$!

# Cleanup function
cleanup() {
    echo ""
    echo "🧹 Stopping server..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
}

trap cleanup EXIT

# Wait for server
echo "   Waiting for server to start..."
sleep 3

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Server failed to start"
    echo "Logs:"
    cat /tmp/realm-server.log
    exit 1
fi

echo "✓ Server is running (PID: $SERVER_PID)"
echo ""

# Run test
echo "🧪 Running Paris generation test..."
echo ""

cd sdks/nodejs-ws
REALM_URL="ws://127.0.0.1:$PORT" timeout 30 node test-paris.js

TEST_RESULT=$?
cd ../..

if [ $TEST_RESULT -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Paris test passed!"
    exit 0
else
    echo ""
    echo "❌ Test failed or timed out"
    echo "Server logs (last 20 lines):"
    tail -20 /tmp/realm-server.log
    exit 1
fi

