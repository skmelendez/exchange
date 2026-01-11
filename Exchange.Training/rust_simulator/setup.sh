#!/bin/bash
# Setup script for EXCHANGE Rust Simulator
# Run this script to install Rust, maturin, and build the Python extension

set -e

echo "=== EXCHANGE Rust Simulator Setup ==="

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo "Rust not found. Installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "Rust found: $(rustc --version)"
fi

# Ensure cargo is in PATH
source "$HOME/.cargo/env" 2>/dev/null || true

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin
else
    echo "maturin found: $(maturin --version)"
fi

# Build the extension in development mode
echo ""
echo "Building Rust extension..."
cd "$(dirname "$0")"
maturin develop --release

echo ""
echo "=== Build Complete! ==="
echo ""
echo "The extension is now installed. Test it with:"
echo "  python -c 'from exchange_simulator import RustSimulator; s = RustSimulator(); print(f\"Moves: {s.num_moves()}\")'"
