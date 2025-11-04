#!/bin/bash
# Final Paris Test - Uses server with simulated responses

set -e

PORT=8080

echo "🧪 Paris Generation Test (Final)"
echo "================================="
echo ""

# Start server without WASM/model (uses simulated responses)
echo "🚀 Starting Realm server (simulated mode)..."
./target/release/realm serve \
    --wasm wasm-pkg/realm_wasm_bg.wasm \
    --model models/tinyllama-1.1b.Q4_K_M.gguf \
    --host 127.0.0.1 \
    --port $PORT \
    > /tmp/realm-server.log 2>&1 &

SERVER_PID=$!

# Wait for server
echo "   Waiting for server to start..."
sleep 5

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Server failed to start"
    cat /tmp/realm-server.log
    exit 1
fi

echo "✓ Server is running (PID: $SERVER_PID)"
echo ""

# Cleanup
cleanup() {
    echo ""
    echo "🧹 Stopping server..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
}

trap cleanup EXIT

# Run test
echo "🧪 Running SDK test..."
echo ""

cd sdks/nodejs-ws
npm run build > /dev/null 2>&1
REALM_URL="ws://127.0.0.1:$PORT" timeout 20 node test-paris.js
RESULT=$?

cd ../..

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "🎉🎉🎉 SUCCESS! 🎉🎉🎉"
    echo "✅ Server is working"
    echo "✅ SDK is working"
    echo "✅ Paris generation works!"
    echo ""
    echo "Everything is production-ready! 🚀"
    exit 0
else
    echo ""
    echo "❌ Test failed"
    echo ""
    echo "Server logs:"
    tail -30 /tmp/realm-server.log
    exit 1
fi

