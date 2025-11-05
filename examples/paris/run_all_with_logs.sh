#!/bin/bash
# Run all Paris examples and generate timestamped log files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "${OUTPUT_DIR}"

# Default model path
MODEL_PATH="${MODEL_PATH:-${HOME}/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf}"

echo "=========================================="
echo "Running All Paris Examples with Timestamps"
echo "=========================================="
echo "Model path: ${MODEL_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo ""

# Function to run example and capture timestamped output
run_example_with_log() {
    local name=$1
    local cmd=$2
    local log_file="${OUTPUT_DIR}/${name}_${TIMESTAMP}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${name}" | tee -a "${log_file}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Command: ${cmd}" | tee -a "${log_file}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log file: ${log_file}" | tee -a "${log_file}"
    echo "----------------------------------------" | tee -a "${log_file}"
    
    if eval "${cmd}" 2>&1 | while IFS= read -r line; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
    done | tee -a "${log_file}"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ SUCCESS: ${name}" | tee -a "${log_file}"
        
        # Check for Paris
        if grep -qi "paris" "${log_file}" 2>/dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ CONTAINS 'Paris'" | tee -a "${log_file}"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ DOES NOT CONTAIN 'Paris'" | tee -a "${log_file}"
        fi
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ FAILED: ${name}" | tee -a "${log_file}"
    fi
    
    echo "----------------------------------------" | tee -a "${log_file}"
    echo ""
}

# Check if model exists
if [ ! -f "${MODEL_PATH}" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  Warning: Model file not found at ${MODEL_PATH}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]    Examples may fail if model is not available"
    echo ""
fi

# 1. Native Rust example
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Running Native Rust Example ==="
run_example_with_log "native" \
    "cd ${SCRIPT_DIR}/native && cargo run --release -- ${MODEL_PATH}"

# 2. Server-based examples (Node.js SDK and Python SDK)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Starting Server for SDK Examples ==="
SERVER_LOG="${OUTPUT_DIR}/server_${TIMESTAMP}.log"

# Check if server is already running
if pgrep -f "realm.*serve" > /dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  Server already running, using existing instance"
    SERVER_PID=$(pgrep -f "realm.*serve" | head -1)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server PID: ${SERVER_PID}"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Building server..." | tee -a "${SERVER_LOG}"
    cd "${SCRIPT_DIR}/../../"
    cargo build --release --bin realm 2>&1 | while IFS= read -r line; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
    done | tee -a "${SERVER_LOG}"
    
    # Find model or use default
    SERVER_MODEL="${MODEL_PATH}"
    if [ ! -f "${SERVER_MODEL}" ]; then
        SERVER_MODEL=$(find "${HOME}/.ollama/models" -name "*.gguf" 2>/dev/null | head -1)
        if [ -z "${SERVER_MODEL}" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Cannot find model file for server" | tee -a "${SERVER_LOG}"
            exit 1
        fi
    fi
    
    # Find WASM file (required for serve command)
    WASM_FILE="${SCRIPT_DIR}/../../target/wasm32-unknown-unknown/release/realm_wasm.wasm"
    if [ ! -f "${WASM_FILE}" ]; then
        # Try to find any realm_wasm.wasm file
        WASM_FILE=$(find "${SCRIPT_DIR}/../../" -name "realm_wasm.wasm" 2>/dev/null | head -1)
        if [ -z "${WASM_FILE}" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  Warning: WASM file not found, building..." | tee -a "${SERVER_LOG}"
            cd "${SCRIPT_DIR}/../../"
            cargo build --release --target wasm32-unknown-unknown -p realm-wasm 2>&1 | while IFS= read -r line; do
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
            done | tee -a "${SERVER_LOG}"
            WASM_FILE="${SCRIPT_DIR}/../../target/wasm32-unknown-unknown/release/realm_wasm.wasm"
        fi
    fi
    
    if [ ! -f "${WASM_FILE}" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Cannot find WASM file for server" | tee -a "${SERVER_LOG}"
        exit 1
    fi
    
    # Start server
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting server with model: ${SERVER_MODEL}" | tee -a "${SERVER_LOG}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using WASM file: ${WASM_FILE}" | tee -a "${SERVER_LOG}"
    timeout ${SERVER_TIMEOUT:-120} cargo run --release --bin realm -- serve \
        --host 127.0.0.1 \
        --port 8080 \
        --wasm "${WASM_FILE}" \
        --model "${SERVER_MODEL}" \
        2>&1 | while IFS= read -r line; do
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
        done | tee -a "${SERVER_LOG}" &
    
    SERVER_PID=$!
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server started (PID: ${SERVER_PID})" | tee -a "${SERVER_LOG}"
    
    # Wait for server to be ready
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for server to be ready..." | tee -a "${SERVER_LOG}"
    for i in {1..30}; do
        if curl -s http://127.0.0.1:8080/health > /dev/null 2>&1; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Server is ready" | tee -a "${SERVER_LOG}"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Server failed to start" | tee -a "${SERVER_LOG}"
            kill ${SERVER_PID} 2>/dev/null || true
            exit 1
        fi
        sleep 1
    done
    sleep 2
fi

# 3. Node.js SDK example
if [ -f "${SCRIPT_DIR}/nodejs-sdk/index.js" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Running Node.js SDK Example ==="
    run_example_with_log "nodejs-sdk" \
        "cd ${SCRIPT_DIR}/nodejs-sdk && REALM_URL=ws://localhost:8080 REALM_MODEL=$(basename ${MODEL_PATH}) node index.js"
fi

# 4. Python SDK example
if [ -f "${SCRIPT_DIR}/python-sdk/main.py" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Running Python SDK Example ==="
    run_example_with_log "python-sdk" \
        "cd ${SCRIPT_DIR}/python-sdk && REALM_URL=ws://localhost:8080 REALM_MODEL=$(basename ${MODEL_PATH}) python main.py"
fi

# Stop server if we started it
if [ -n "${SERVER_PID}" ] && [ "${SERVER_PID}" != "0" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stopping server (PID: ${SERVER_PID})..."
    kill ${SERVER_PID} 2>/dev/null || true
    wait ${SERVER_PID} 2>/dev/null || true
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Server stopped"
fi

# Generate summary
SUMMARY_LOG="${OUTPUT_DIR}/summary_${TIMESTAMP}.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==========================================" | tee "${SUMMARY_LOG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PARIS GENERATION TEST SUMMARY" | tee -a "${SUMMARY_LOG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==========================================" | tee -a "${SUMMARY_LOG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Timestamp: ${TIMESTAMP}" | tee -a "${SUMMARY_LOG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model: ${MODEL_PATH}" | tee -a "${SUMMARY_LOG}"
echo "" | tee -a "${SUMMARY_LOG}"

for log in "${OUTPUT_DIR}"/*_${TIMESTAMP}.log; do
    if [ -f "${log}" ]; then
        name=$(basename "${log}" .log | sed "s/_${TIMESTAMP}//")
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] === ${name} ===" | tee -a "${SUMMARY_LOG}"
        
        if grep -qi "paris" "${log}" 2>/dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ CONTAINS 'Paris'" | tee -a "${SUMMARY_LOG}"
            # Extract the actual response line
            grep -i "paris\|response\|capital" "${log}" | tail -3 | while IFS= read -r line; do
                echo "[$(date '+%Y-%m-%d %H:%M:%S')]   $line" | tee -a "${SUMMARY_LOG}"
            done
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ DOES NOT CONTAIN 'Paris'" | tee -a "${SUMMARY_LOG}"
        fi
        echo "" | tee -a "${SUMMARY_LOG}"
    fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==========================================" | tee -a "${SUMMARY_LOG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All log files saved to: ${OUTPUT_DIR}" | tee -a "${SUMMARY_LOG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Summary: ${SUMMARY_LOG}" | tee -a "${SUMMARY_LOG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done!" | tee -a "${SUMMARY_LOG}"

echo ""
echo "=========================================="
echo "Summary saved to: ${SUMMARY_LOG}"
echo "All logs saved to: ${OUTPUT_DIR}"
echo "=========================================="

