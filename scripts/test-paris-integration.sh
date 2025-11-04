#!/bin/bash
# Full Integration Test: Paris Generation
# Tests server + SDK end-to-end

set -e

echo "🧪 Full Integration Test: Paris Generation"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo "1️⃣  Checking prerequisites..."

# Check for server binary
if [ ! -f "target/debug/realm" ] && [ ! -f "target/release/realm" ]; then
    echo -e "${YELLOW}⚠️  Server binary not found. Building...${NC}"
    cargo build --bin realm
fi

SERVER_BIN="target/debug/realm"
if [ -f "target/release/realm" ]; then
    SERVER_BIN="target/release/realm"
fi
echo -e "${GREEN}✅ Server binary: $SERVER_BIN${NC}"

# Check for WASM
WASM_FILE=""
if [ -f "crates/realm-wasm/pkg/realm_wasm_bg.wasm" ]; then
    WASM_FILE="crates/realm-wasm/pkg/realm_wasm_bg.wasm"
elif [ -f "target/wasm32-unknown-unknown/release/realm_wasm.wasm" ]; then
    WASM_FILE="target/wasm32-unknown-unknown/release/realm_wasm.wasm"
else
    echo -e "${RED}❌ WASM file not found${NC}"
    echo "   Looking for: crates/realm-wasm/pkg/realm_wasm_bg.wasm"
    exit 1
fi
echo -e "${GREEN}✅ WASM file: $WASM_FILE${NC}"

# Check for model
MODEL_FILE=""
if [ -f "models/tinyllama-1.1b.Q4_K_M.gguf" ]; then
    MODEL_FILE="models/tinyllama-1.1b.Q4_K_M.gguf"
elif [ -f "models/llama-2-7b-chat-q4_k_m.gguf" ]; then
    MODEL_FILE="models/llama-2-7b-chat-q4_k_m.gguf"
else
    echo -e "${YELLOW}⚠️  Model file not found in models/ directory${NC}"
    echo "   Using simulated responses (server will work but output may not be 'Paris')"
    MODEL_FILE=""
fi

if [ -n "$MODEL_FILE" ]; then
    echo -e "${GREEN}✅ Model file: $MODEL_FILE${NC}"
fi

# Check Node.js SDK
if [ ! -d "sdks/nodejs-ws/dist" ]; then
    echo -e "${YELLOW}⚠️  Building Node.js SDK...${NC}"
    cd sdks/nodejs-ws
    npm install
    npm run build
    cd ../..
fi
echo -e "${GREEN}✅ Node.js SDK ready${NC}"

echo ""
echo "2️⃣  Starting Realm server..."
echo ""

# Start server in background
if [ -n "$MODEL_FILE" ]; then
    $SERVER_BIN serve \
        --wasm "$WASM_FILE" \
        --model "$MODEL_FILE" \
        --host 127.0.0.1 \
        --port 8080 \
        > /tmp/realm-server.log 2>&1 &
else
    # Without model, server will use simulated responses
    $SERVER_BIN serve \
        --wasm "$WASM_FILE" \
        --host 127.0.0.1 \
        --port 8080 \
        > /tmp/realm-server.log 2>&1 &
fi

SERVER_PID=$!
echo "   Server PID: $SERVER_PID"

# Wait for server to start
echo "   Waiting for server to start..."
sleep 3

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}❌ Server failed to start${NC}"
    echo "   Check logs: /tmp/realm-server.log"
    cat /tmp/realm-server.log
    exit 1
fi

echo -e "${GREEN}✅ Server is running${NC}"
echo ""

# Run SDK test
echo "3️⃣  Running SDK test..."
echo ""

cd sdks/nodejs-ws
node test-paris.js
TEST_RESULT=$?
cd ../..

echo ""

# Cleanup
echo "4️⃣  Cleaning up..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo -e "${GREEN}✅ Server stopped${NC}"

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}🎉 Integration test PASSED!${NC}"
    exit 0
else
    echo -e "${RED}❌ Integration test FAILED${NC}"
    exit 1
fi

