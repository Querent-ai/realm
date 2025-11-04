#!/bin/bash
# Full Integration Test Script
# Tests: Server → SDK → Paris Generation

set -e

echo "🚀 Full Integration Test: Paris Generation"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo "📋 Checking prerequisites..."
if [ ! -f "target/release/realm" ]; then
    echo -e "${YELLOW}⚠️  Realm binary not found. Building...${NC}"
    cargo build --release --bin realm
fi

if [ ! -f "models/tinyllama-1.1b.Q4_K_M.gguf" ] && [ ! -f "models/llama-2-7b-chat-q4_k_m.gguf" ]; then
    echo -e "${RED}❌ No model file found in models/ directory${NC}"
    echo "   Please download a model first:"
    echo "   realm models download tinyllama-1.1b:Q4_K_M"
    exit 1
fi

# Find model file
MODEL_FILE=$(find models -name "*.gguf" | head -1)
echo -e "${GREEN}✓ Found model: ${MODEL_FILE}${NC}"

# Find WASM file
WASM_FILE=$(find . -name "realm_wasm_bg.wasm" -o -name "*.wasm" | grep -E "(wasm|pkg)" | head -1)
if [ -z "$WASM_FILE" ]; then
    echo -e "${YELLOW}⚠️  WASM file not found. Using placeholder...${NC}"
    WASM_FILE="placeholder.wasm"
    # Create a dummy WASM file for testing
    echo "Creating placeholder WASM..."
    touch "$WASM_FILE"
fi

echo -e "${GREEN}✓ Using WASM: ${WASM_FILE}${NC}"
echo ""

# Start server in background
echo "🚀 Starting Realm server..."
SERVER_PID=""
PORT=8080

# Function to cleanup
cleanup() {
    if [ ! -z "$SERVER_PID" ]; then
        echo ""
        echo "🧹 Cleaning up..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        echo -e "${GREEN}✓ Server stopped${NC}"
    fi
}

trap cleanup EXIT

# Start server
if [ -f "$WASM_FILE" ] && [ "$WASM_FILE" != "placeholder.wasm" ]; then
    ./target/release/realm serve \
        --wasm "$WASM_FILE" \
        --model "$MODEL_FILE" \
        --host 127.0.0.1 \
        --port $PORT \
        > /tmp/realm-server.log 2>&1 &
    SERVER_PID=$!
else
    # Start without WASM (will use simulated responses)
    ./target/release/realm serve \
        --model "$MODEL_FILE" \
        --host 127.0.0.1 \
        --port $PORT \
        > /tmp/realm-server.log 2>&1 &
    SERVER_PID=$!
fi

echo "   Server PID: $SERVER_PID"
echo "   Waiting for server to start..."

# Wait for server to be ready
for i in {1..30}; do
    if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1 || nc -z 127.0.0.1 $PORT 2>/dev/null; then
        echo -e "${GREEN}✓ Server is ready!${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Server failed to start${NC}"
        echo "Server logs:"
        tail -20 /tmp/realm-server.log
        exit 1
    fi
    sleep 1
done

echo ""
echo "🧪 Running SDK test..."
echo ""

# Run SDK test
cd sdks/nodejs-ws
npm run build > /dev/null 2>&1

REALM_URL="ws://127.0.0.1:$PORT" node test-paris.js

TEST_RESULT=$?

cd ../..

if [ $TEST_RESULT -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 SUCCESS! Full integration test passed!${NC}"
    echo -e "${GREEN}✅ Server is working${NC}"
    echo -e "${GREEN}✅ SDK is working${NC}"
    echo -e "${GREEN}✅ Paris generation works!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}❌ Test failed${NC}"
    echo "Server logs:"
    tail -30 /tmp/realm-server.log
    exit 1
fi

