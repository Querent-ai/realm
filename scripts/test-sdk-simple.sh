#!/bin/bash
# Simple SDK Test - Using server with simulated responses
# This tests the SDK connection without requiring actual WASM/model

set -e

echo "🧪 Simple SDK Test: Paris Generation"
echo "====================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Build SDK if needed
echo "1️⃣  Building Node.js SDK..."
cd sdks/nodejs-ws
if [ ! -d "dist" ]; then
    npm install
    npm run build
fi
echo -e "${GREEN}✅ SDK ready${NC}"
cd ../..

# Start server WITHOUT WASM (will use simulated responses)
echo ""
echo "2️⃣  Starting server (simulated mode)..."
echo ""

# Use realm CLI with dummy files
# The server will use simulated responses if WASM fails
DUMMY_WASM="/tmp/dummy.wasm"
DUMMY_MODEL="/tmp/dummy.gguf"
echo "dummy" > $DUMMY_WASM
echo "dummy" > $DUMMY_MODEL

# Start server - it will fail to load WASM but use simulated responses
./target/release/realm serve \
    --wasm $DUMMY_WASM \
    --model $DUMMY_MODEL \
    --host 127.0.0.1 \
    --port 8080 \
    > /tmp/realm-server.log 2>&1 &
SERVER_PID=$!

echo "   Server PID: $SERVER_PID"
echo "   Waiting for server to start..."
sleep 5

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}❌ Server failed to start${NC}"
    echo "   Logs:"
    tail -20 /tmp/realm-server.log
    exit 1
fi

echo -e "${GREEN}✅ Server is running${NC}"
echo ""

# Test SDK
echo "3️⃣  Testing SDK connection..."
echo ""

cd sdks/nodejs-ws
node test-paris.js
TEST_RESULT=$?
cd ../..

# Cleanup
echo ""
echo "4️⃣  Cleaning up..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
sleep 1
echo -e "${GREEN}✅ Server stopped${NC}"

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}🎉 SDK Test PASSED!${NC}"
    exit 0
else
    echo -e "${RED}❌ SDK Test FAILED${NC}"
    exit 1
fi

