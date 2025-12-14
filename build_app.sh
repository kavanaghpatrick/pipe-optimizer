#!/bin/bash
# Build script for Pipe Optimizer Mac app
# Creates a standalone .app bundle that anyone can double-click

set -e

echo "=========================================="
echo "Building Pipe Optimizer Mac App"
echo "=========================================="

# Clean previous builds
rm -rf build dist

# Build using the spec file
pyinstaller PipeOptimizer.spec --clean --noconfirm

# Fix CBC solver permissions (PyInstaller may strip execute bits)
echo "Fixing solver permissions..."
# Resources location
chmod +x "dist/Pipe Optimizer.app/Contents/Resources/pulp/solverdir/cbc/osx/i64/cbc" 2>/dev/null || true
chmod +x "dist/Pipe Optimizer.app/Contents/Resources/pulp/solverdir/cbc/linux/i64/cbc" 2>/dev/null || true
chmod +x "dist/Pipe Optimizer.app/Contents/Resources/pulp/solverdir/cbc/linux/arm64/cbc" 2>/dev/null || true
# Frameworks location (symlinks may point here)
chmod +x "dist/Pipe Optimizer.app/Contents/Frameworks/pulp/solverdir/cbc/osx/i64/cbc" 2>/dev/null || true
chmod +x "dist/Pipe Optimizer.app/Contents/Frameworks/pulp/solverdir/cbc/linux/i64/cbc" 2>/dev/null || true
chmod +x "dist/Pipe Optimizer.app/Contents/Frameworks/pulp/solverdir/cbc/linux/arm64/cbc" 2>/dev/null || true

# Create apis directory for pulp path resolution (pulp looks for apis/../solverdir)
mkdir -p "dist/Pipe Optimizer.app/Contents/Frameworks/pulp/apis" 2>/dev/null || true
mkdir -p "dist/Pipe Optimizer.app/Contents/Resources/pulp/apis" 2>/dev/null || true

echo ""
echo "=========================================="
echo "BUILD COMPLETE!"
echo "=========================================="
echo ""
echo "Your app is at: dist/Pipe Optimizer.app"
echo ""
echo "To distribute:"
echo "  1. Copy 'dist/Pipe Optimizer.app' to the user"
echo "  2. They can drag it to Applications folder"
echo "  3. Double-click to run!"
echo ""
echo "Note: First launch may require right-click > Open"
echo "      (macOS Gatekeeper for unsigned apps)"
