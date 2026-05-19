#!/bin/bash
# ============================================================
#  build_android.sh -- ndk-build for RK3568 Android
#  Usage: cd deploy_cpp && bash scripts/build_android.sh
# ============================================================
set -e

if [ -z "$NDK_HOME" ]; then
    echo "[ERROR] NDK_HOME not set."
    echo "  export NDK_HOME=~/Android/Sdk/ndk/25.2.9519653"
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
JNI_DIR="$PROJECT_DIR/jni"

cd "$JNI_DIR"
"$NDK_HOME/ndk-build" NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=Android.mk NDK_APPLICATION_MK=Application.mk

OUT_DIR="$PROJECT_DIR/output/arm64-v8a"
mkdir -p "$OUT_DIR"
cp "$PROJECT_DIR/libs/arm64-v8a/libexcavator_algo.so" "$OUT_DIR/"
cp "$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" "$OUT_DIR/" 2>/dev/null || true

echo ""
echo "============================================================"
echo "  Build complete: $OUT_DIR/libexcavator_algo.so"
echo "============================================================"
