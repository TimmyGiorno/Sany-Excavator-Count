#!/bin/bash
# ============================================================
#  build_linux.sh -- Linux ARM / x86 build
#  Usage: bash scripts/build_linux.sh [rknn|mock|onnx]
# ============================================================
set -e

BACKEND="${1:-mock}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build_linux"

case "$BACKEND" in
    rknn)  FLAGS="-DUSE_RKNN=ON -DUSE_MOCK=OFF -DUSE_ONNX=OFF" ;;
    onnx)  FLAGS="-DUSE_ONNX=ON -DUSE_MOCK=OFF -DUSE_RKNN=OFF" ;;
    mock)  FLAGS="-DUSE_MOCK=ON -DUSE_RKNN=OFF -DUSE_ONNX=OFF" ;;
    *)     echo "Unknown backend: $BACKEND (use: rknn|mock|onnx)"; exit 1 ;;
esac

echo "[INFO] Configuring CMake (backend: $BACKEND)..."
cmake -B "$BUILD_DIR" -S "$PROJECT_DIR" $FLAGS -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j$(nproc)

echo ""
echo "============================================================"
echo "  Build complete: $BUILD_DIR/test_video"
echo "============================================================"
