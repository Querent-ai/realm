#!/bin/bash
# Build WASM bindings with wasm-bindgen

set -e

echo "🔨 Building WASM bindings for realm-wasm..."
echo ""

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack not found. Installing..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Build WASM module first
echo "📦 Building WASM module..."
cargo build -p realm-wasm --target wasm32-unknown-unknown --release || {
    echo "❌ WASM build failed"
    exit 1
}

# Generate wasm-bindgen bindings
echo "🔧 Generating wasm-bindgen bindings..."
cd crates/realm-wasm

wasm-pack build --target nodejs --release --out-dir ../../pkg || {
    echo "❌ wasm-pack build failed"
    echo ""
    echo "Note: Host functions need to be provided at runtime."
    echo "See examples/js-paris-generation/bridge.js for bridge implementation."
    exit 1
}

cd ../..

echo ""
echo "✅ WASM bindings generated in pkg/"
echo ""
echo "⚠️  IMPORTANT: Host functions must be provided at runtime."
echo "   The WASM module expects these imports:"
echo "   - realm_store_model"
echo "   - realm_get_tensor"
echo "   - realm_get_model_info"
echo "   - realm_remove_model"
echo ""
echo "   Use bridge.js or implement in your runtime."

