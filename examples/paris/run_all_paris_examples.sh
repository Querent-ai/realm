#!/bin/bash
# Run all Paris examples and capture outputs to .log files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${LOG_DIR}"

# Default model path - can be overridden with MODEL_PATH env var
MODEL_PATH="${MODEL_PATH:-${HOME}/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf}"

echo "=========================================="
echo "Running All Paris Examples"
echo "=========================================="
echo "Model path: ${MODEL_PATH}"
echo "Output directory: ${LOG_DIR}"
echo ""

# Check if model exists
if [ ! -f "${MODEL_PATH}" ]; then
    echo "⚠️  Warning: Model file not found at ${MODEL_PATH}"
    echo "   Examples may fail if model is not available"
    echo ""
fi

# Function to run example and capture output
run_example() {
    local name=$1
    local cmd=$2
    local log_file="${LOG_DIR}/${name}.log"
    
    echo "Running: ${name}..."
    echo "  Command: ${cmd}"
    echo "  Log: ${log_file}"
    
    if eval "${cmd}" > "${log_file}" 2>&1; then
        echo "  ✅ Success"
    else
        echo "  ❌ Failed (check ${log_file})"
    fi
    echo ""
}

# 1. Native Rust example
run_example "native" \
    "cd ${SCRIPT_DIR}/native && cargo run --release -- ${MODEL_PATH}"

# 2. Paris generation (old example)
run_example "paris-generation" \
    "cd ${SCRIPT_DIR}/../paris-generation && cargo run --release -- ${MODEL_PATH}"

# 3. Node.js WASM example (if WASM is built)
if [ -f "${SCRIPT_DIR}/../../crates/realm-wasm/pkg/realm_wasm.js" ]; then
    run_example "nodejs-wasm" \
        "cd ${SCRIPT_DIR}/nodejs-wasm && node index.js ${MODEL_PATH}"
else
    echo "⚠️  Skipping nodejs-wasm: WASM not built"
    echo "   Build with: cd crates/realm-wasm && wasm-pack build --target nodejs"
    echo ""
fi

# 4. Server-based examples (Node.js SDK and Python SDK)
# First, check if server is already running
if ! pgrep -f "realm-server" > /dev/null; then
    echo "Starting Realm server for SDK examples..."
    SERVER_LOG="${LOG_DIR}/server.log"
    
    # Start server in background
    cd "${SCRIPT_DIR}/../../"
    cargo build --release --bin realm-server 2>&1 | tee -a "${SERVER_LOG}"
    
    # Find model or use default
    SERVER_MODEL="${MODEL_PATH}"
    if [ ! -f "${SERVER_MODEL}" ]; then
        # Try to find any .gguf file
        SERVER_MODEL=$(find "${HOME}/.ollama/models" -name "*.gguf" 2>/dev/null | head -1)
        if [ -z "${SERVER_MODEL}" ]; then
            echo "❌ Cannot find model file for server"
            echo "   Set MODEL_PATH environment variable"
            exit 1
        fi
    fi
    
    # Start server
    timeout 120 cargo run --release --bin realm-server -- \
        --host 127.0.0.1 \
        --port 8080 \
        --model "${SERVER_MODEL}" \
        --wasm "${SCRIPT_DIR}/../../target/wasm32-unknown-unknown/release/realm_wasm.wasm" \
        > "${SERVER_LOG}" 2>&1 &
    
    SERVER_PID=$!
    echo "Server started (PID: ${SERVER_PID})"
    
    # Wait for server to be ready
    echo "Waiting for server to be ready..."
    for i in {1..30}; do
        if curl -s http://127.0.0.1:8080/health > /dev/null 2>&1; then
            echo "✅ Server is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "❌ Server failed to start"
            kill ${SERVER_PID} 2>/dev/null || true
            exit 1
        fi
        sleep 1
    done
    sleep 2  # Give server a moment to fully initialize
    
    # Run Node.js SDK example
    if [ -f "${SCRIPT_DIR}/nodejs-sdk/index.js" ]; then
        run_example "nodejs-sdk" \
            "cd ${SCRIPT_DIR}/nodejs-sdk && REALM_URL=ws://localhost:8080 REALM_MODEL=$(basename ${SERVER_MODEL}) node index.js"
    fi
    
    # Run Python SDK example
    if [ -f "${SCRIPT_DIR}/python-sdk/main.py" ]; then
        run_example "python-sdk" \
            "cd ${SCRIPT_DIR}/python-sdk && REALM_URL=ws://localhost:8080 REALM_MODEL=$(basename ${SERVER_MODEL}) python main.py"
    fi
    
    # Stop server
    echo "Stopping server..."
    kill ${SERVER_PID} 2>/dev/null || true
    wait ${SERVER_PID} 2>/dev/null || true
    echo "✅ Server stopped"
    echo ""
else
    echo "⚠️  Server already running, skipping SDK examples"
    echo ""
fi

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Log files saved to: ${LOG_DIR}"
echo ""
for log in "${LOG_DIR}"/*.log; do
    if [ -f "${log}" ]; then
        name=$(basename "${log}" .log)
        if grep -q "Paris" "${log}" 2>/dev/null; then
            echo "✅ ${name}: Contains 'Paris'"
        else
            echo "❌ ${name}: Does not contain 'Paris'"
        fi
    fi
done
echo ""
echo "Done!"

