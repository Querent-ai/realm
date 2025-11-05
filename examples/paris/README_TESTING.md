# Paris Generation Testing Guide

## Running All Tests with Timestamped Logs

To generate timestamped log files for all Paris examples:

```bash
cd examples/paris
./run_all_with_logs.sh
```

This will:
1. Run all Paris examples (Native, Node.js SDK, Python SDK)
2. Generate timestamped log files in `outputs/logs/`
3. Create a summary log showing which examples produced "Paris"

## Log Files Generated

Each test run creates timestamped log files:
- `native_YYYYMMDD_HHMMSS.log` - Native Rust example output
- `nodejs-sdk_YYYYMMDD_HHMMSS.log` - Node.js SDK example output
- `python-sdk_YYYYMMDD_HHMMSS.log` - Python SDK example output
- `server_YYYYMMDD_HHMMSS.log` - Server logs (if started)
- `summary_YYYYMMDD_HHMMSS.log` - Summary of all tests

## Verifying Results

Each log file contains:
- Timestamp for each line
- Full output from the example
- Verification status (✅ CONTAINS 'Paris' or ❌ DOES NOT CONTAIN 'Paris')

## Example Output

```
[2025-01-31 14:30:15] Starting: native
[2025-01-31 14:30:15] Command: cd examples/paris/native && cargo run --release -- <model>
[2025-01-31 14:30:45] Response: The capital of France is Paris.
[2025-01-31 14:30:45] ✅ SUCCESS: native
[2025-01-31 14:30:45] ✅ CONTAINS 'Paris'
```

## Manual Testing

### Native Rust Example
```bash
cd examples/paris/native
cargo run --release -- ~/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf 2>&1 | \
    while IFS= read -r line; do echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"; done | \
    tee paris_native_$(date +%Y%m%d_%H%M%S).log
```

### Node.js SDK Example
```bash
# Terminal 1: Start server
cargo run --release --bin realm -- server --host 127.0.0.1 --port 8080 --model <model>

# Terminal 2: Run example
cd examples/paris/nodejs-sdk
REALM_URL=ws://localhost:8080 REALM_MODEL=<model> node index.js 2>&1 | \
    while IFS= read -r line; do echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"; done | \
    tee paris_nodejs_$(date +%Y%m%d_%H%M%S).log
```

### Python SDK Example
```bash
# Terminal 1: Start server (same as above)

# Terminal 2: Run example
cd examples/paris/python-sdk
REALM_URL=ws://localhost:8080 REALM_MODEL=<model> python main.py 2>&1 | \
    while IFS= read -r line; do echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"; done | \
    tee paris_python_$(date +%Y%m%d_%H%M%S).log
```

## Environment Variables

- `MODEL_PATH` - Path to model file (default: `~/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf`)
- `SERVER_TIMEOUT` - Server timeout in seconds (default: 120)
- `REALM_URL` - Server URL for SDK examples (default: `ws://localhost:8080`)
- `REALM_MODEL` - Model name for SDK examples

