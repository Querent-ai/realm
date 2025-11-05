#!/bin/bash
# Comprehensive verification script for all Paris generation stacks
# This ensures all stacks work and produce "Paris" before check-in

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/verification"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "${OUTPUT_DIR}"

# Default model path
MODEL_PATH="${MODEL_PATH:-${HOME}/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf}"

echo "==========================================" | tee "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
echo "Comprehensive Stack Verification" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
echo "==========================================" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
echo "Timestamp: ${TIMESTAMP}" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
echo "Model path: ${MODEL_PATH}" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
echo "" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"

PASSED=0
FAILED=0

# Function to test and log result
test_stack() {
    local name=$1
    local test_cmd=$2
    local log_file="${OUTPUT_DIR}/${name}_${TIMESTAMP}.log"
    
    echo "" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
    echo "=== Testing ${name} ===" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
    echo "Command: ${test_cmd}" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
    
    if eval "${test_cmd}" > "${log_file}" 2>&1; then
        # Check if output contains "Paris" (case-insensitive)
        if grep -qi "paris" "${log_file}" 2>/dev/null; then
            echo "✅ ${name}: PASSED - Produces 'Paris'" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
            ((PASSED++))
            return 0
        else
            echo "❌ ${name}: FAILED - Does not produce 'Paris'" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
            echo "   Last 10 lines of output:" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
            tail -10 "${log_file}" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
            ((FAILED++))
            return 1
        fi
    else
        echo "❌ ${name}: FAILED - Command failed" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
        echo "   Last 10 lines of error:" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
        tail -10 "${log_file}" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
        ((FAILED++))
        return 1
    fi
}

# 1. Compilation Check
echo "=== Step 1: Compilation Check ===" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
cd "${REPO_ROOT}"

if cargo check --workspace 2>&1 | grep -q "Finished"; then
    echo "✅ All crates compile successfully" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
    ((PASSED++))
else
    echo "❌ Compilation failed" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
    ((FAILED++))
    exit 1
fi

# 2. Native Rust Example
echo "" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
echo "=== Step 2: Native Rust Stack ===" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
if [ -f "${MODEL_PATH}" ]; then
    test_stack "native" \
        "cd ${SCRIPT_DIR}/native && cargo run --release -- ${MODEL_PATH} 'What is the capital of France?'"
else
    echo "⚠️  Skipping native test: Model not found at ${MODEL_PATH}" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
fi

# 3. Build Server
echo "" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
echo "=== Step 3: Building Server ===" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
cd "${REPO_ROOT}"

if cargo build --release --bin realm 2>&1 | grep -q "Finished"; then
    echo "✅ Server built successfully" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
    ((PASSED++))
else
    echo "❌ Server build failed" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
    ((FAILED++))
fi

# 4. Build WASM
echo "" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
echo "=== Step 4: Building WASM ===" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
if cargo build --release --target wasm32-unknown-unknown -p realm-wasm 2>&1 | grep -q "Finished"; then
    echo "✅ WASM built successfully" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
    ((PASSED++))
else
    echo "❌ WASM build failed" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
    ((FAILED++))
fi

# 5. Node.js SDK (requires server)
if [ -f "${MODEL_PATH}" ] && [ -f "${SCRIPT_DIR}/nodejs-sdk/index.js" ]; then
    echo "" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
    echo "=== Step 5: Node.js SDK Stack ===" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
    
    # Find WASM file
    WASM_FILE="${REPO_ROOT}/target/wasm32-unknown-unknown/release/realm_wasm.wasm"
    if [ ! -f "${WASM_FILE}" ]; then
        WASM_FILE=$(find "${REPO_ROOT}" -name "realm_wasm.wasm" 2>/dev/null | head -1)
    fi
    
    if [ -f "${WASM_FILE}" ]; then
        # Start server in background
        SERVER_LOG="${OUTPUT_DIR}/server_${TIMESTAMP}.log"
        echo "Starting server..." | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
        timeout 120 "${REPO_ROOT}/target/release/realm" serve \
            --host 127.0.0.1 \
            --port 8080 \
            --wasm "${WASM_FILE}" \
            --model "${MODEL_PATH}" \
            > "${SERVER_LOG}" 2>&1 &
        SERVER_PID=$!
        
        # Wait for server to be ready
        echo "Waiting for server to be ready..." | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
        for i in {1..30}; do
            if curl -s http://127.0.0.1:8080/health > /dev/null 2>&1; then
                echo "✅ Server is ready" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
                break
            fi
            if [ $i -eq 30 ]; then
                echo "❌ Server failed to start" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
                kill ${SERVER_PID} 2>/dev/null || true
                ((FAILED++))
                break
            fi
            sleep 1
        done
        
        sleep 2
        
        # Test Node.js SDK
        if [ -n "${SERVER_PID}" ] && kill -0 ${SERVER_PID} 2>/dev/null; then
            test_stack "nodejs-sdk" \
                "cd ${SCRIPT_DIR}/nodejs-sdk && REALM_URL=ws://localhost:8080 REALM_MODEL=$(basename ${MODEL_PATH}) node index.js"
            
            # Stop server
            kill ${SERVER_PID} 2>/dev/null || true
            wait ${SERVER_PID} 2>/dev/null || true
        fi
    else
        echo "⚠️  Skipping Node.js SDK test: WASM file not found" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
    fi
fi

# 6. Python SDK (requires server)
if [ -f "${MODEL_PATH}" ] && [ -f "${SCRIPT_DIR}/python-sdk/main.py" ]; then
    echo "" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
    echo "=== Step 6: Python SDK Stack ===" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
    
    # Find WASM file
    WASM_FILE="${REPO_ROOT}/target/wasm32-unknown-unknown/release/realm_wasm.wasm"
    if [ ! -f "${WASM_FILE}" ]; then
        WASM_FILE=$(find "${REPO_ROOT}" -name "realm_wasm.wasm" 2>/dev/null | head -1)
    fi
    
    if [ -f "${WASM_FILE}" ]; then
        # Start server in background
        SERVER_LOG="${OUTPUT_DIR}/server_python_${TIMESTAMP}.log"
        echo "Starting server..." | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
        timeout 120 "${REPO_ROOT}/target/release/realm" serve \
            --host 127.0.0.1 \
            --port 8081 \
            --wasm "${WASM_FILE}" \
            --model "${MODEL_PATH}" \
            > "${SERVER_LOG}" 2>&1 &
        SERVER_PID=$!
        
        # Wait for server to be ready
        echo "Waiting for server to be ready..." | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
        for i in {1..30}; do
            if curl -s http://127.0.0.1:8081/health > /dev/null 2>&1; then
                echo "✅ Server is ready" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
                break
            fi
            if [ $i -eq 30 ]; then
                echo "❌ Server failed to start" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
                kill ${SERVER_PID} 2>/dev/null || true
                ((FAILED++))
                break
            fi
            sleep 1
        done
        
        sleep 2
        
        # Test Python SDK
        if [ -n "${SERVER_PID}" ] && kill -0 ${SERVER_PID} 2>/dev/null; then
            test_stack "python-sdk" \
                "cd ${SCRIPT_DIR}/python-sdk && REALM_URL=ws://localhost:8081 REALM_MODEL=$(basename ${MODEL_PATH}) python3 main.py"
            
            # Stop server
            kill ${SERVER_PID} 2>/dev/null || true
            wait ${SERVER_PID} 2>/dev/null || true
        fi
    else
        echo "⚠️  Skipping Python SDK test: WASM file not found" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
    fi
fi

# Summary
echo "" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
echo "==========================================" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
echo "Verification Summary" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
echo "==========================================" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
echo "Passed: ${PASSED}" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
echo "Failed: ${FAILED}" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
echo "" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"

if [ ${FAILED} -eq 0 ]; then
    echo "✅ ALL TESTS PASSED - Ready for check-in!" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
    exit 0
else
    echo "❌ SOME TESTS FAILED - Review errors above" | tee -a "${OUTPUT_DIR}/verification_${TIMESTAMP}.log"
    exit 1
fi



