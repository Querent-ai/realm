#!/bin/bash
# Build script for realm-node native addon

set -e

echo "🔨 Building realm-node native addon..."

# Check if neon-cli is installed
if ! command -v neon &> /dev/null; then
    echo "Installing neon-cli..."
    npm install -g neon-cli
fi

# Build native addon
cd "$(dirname "$0")"
npm install
neon build --release

echo "✅ Native addon built: native.node"

