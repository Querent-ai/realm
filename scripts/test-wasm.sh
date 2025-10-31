#!/bin/bash
# Test WASM Paris Generation Example

set -e

echo "🔨 Building WASM module..."
cd crates/realm-wasm
wasm-pack build --target web

echo ""
echo "✅ WASM build complete!"
echo ""
echo "📦 Package contents:"
ls -lh pkg/ | grep -E "\.wasm|\.js"

echo ""
echo "🧪 Running JavaScript test..."
cd ../../examples/paris-generation/js
node index.js

echo ""
echo "✅ Test complete!"
