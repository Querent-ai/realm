#!/bin/bash
# Simple Paris Test - Uses example server

set -e

PORT=8080

echo "🧪 Paris Generation Test"
echo "========================"
echo ""

# Build example server
echo "📦 Building example server..."
cargo build --release --example websocket-server > /dev/null 2>&1

# Start server
echo "🚀 Starting server..."
./target/release/examples/websocket-server > /tmp/realm-server.log 2>&1 &
SERVER_PID=$!

# Wait for server
echo "   Waiting for server..."
sleep 3

# Cleanup
cleanup() {
    kill $SERVER_PID 2>/dev/null || true
}

trap cleanup EXIT

# Run test
echo "🧪 Testing SDK..."
echo ""

cd sdks/nodejs-ws
npm run build > /dev/null 2>&1
REALM_URL="ws://127.0.0.1:$PORT" node test-paris.js
RESULT=$?

cd ../..

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Paris test passed!"
    exit 0
else
    echo ""
    echo "❌ Test failed"
    tail -20 /tmp/realm-server.log
    exit 1
fi

