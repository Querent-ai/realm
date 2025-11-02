#!/bin/bash
# Build complete Node.js SDK
# Builds: realm-node native addon + realm-wasm bindings

set -e

echo "🔨 Building Realm Node.js SDK"
echo "================================"
echo ""

# Step 1: Build native addon
echo "📦 Step 1: Building native addon (realm-node)..."
cd crates/realm-node

if ! command -v neon &> /dev/null; then
    echo "   Installing neon-cli..."
    npm install -g neon-cli || {
        echo "❌ Failed to install neon-cli"
        echo "   Install manually: npm install -g neon-cli"
        exit 1
    }
fi

npm install
neon build --release || {
    echo "❌ Native addon build failed"
    exit 1
}

echo "✅ Native addon built: native.node"
echo ""

# Step 2: Generate WASM bindings
echo "📦 Step 2: Generating WASM bindings..."
cd ../realm-wasm

if ! command -v wasm-pack &> /dev/null; then
    echo "   Installing wasm-pack..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh || {
        echo "❌ Failed to install wasm-pack"
        exit 1
    }
fi

# Build WASM first
cargo build --target wasm32-unknown-unknown --release || {
    echo "❌ WASM build failed"
    exit 1
}

# Generate bindings
wasm-pack build --target nodejs --release --out-dir ../../sdks/nodejs/pkg || {
    echo "❌ wasm-pack build failed"
    exit 1
}

echo "✅ WASM bindings generated: sdks/nodejs/pkg/"
echo ""

# Step 3: Copy native addon to SDK
echo "📦 Step 3: Copying native addon to SDK..."
cd ../..
cp crates/realm-node/native.node sdks/nodejs/ 2>/dev/null || {
    echo "⚠️  Note: native.node location may vary"
    echo "   Find it in: crates/realm-node/target/release/"
}

echo ""
echo "✅ SDK build complete!"
echo ""
echo "🚀 To test:"
echo "   cd sdks/nodejs"
echo "   node test-paris.js ~/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf"
echo ""

