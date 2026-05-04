#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE_DIR="$SCRIPT_DIR"
NDK="/home/baiha/trae/workspace/chese/android-sdk/ndk/27.0.12077973"
JNI_DIR="$SCRIPT_DIR/../composeApp/src/androidMain/jniLibs"

mkdir -p "$JNI_DIR/arm64-v8a"
mkdir -p "$JNI_DIR/armeabi-v7a"
mkdir -p "$JNI_DIR/x86_64"
mkdir -p "$JNI_DIR/x86"

echo "=== Building Rust engine for Android ==="

export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android33-clang"
export CARGO_TARGET_ARMV7_LINUX_ANDROIDEABI_LINKER="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi33-clang"
export CARGO_TARGET_X86_64_LINUX_ANDROID_LINKER="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android33-clang"
export CARGO_TARGET_I686_LINUX_ANDROID_LINKER="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/i686-linux-android33-clang"

echo "--- arm64-v8a ---"
cd "$ENGINE_DIR" && cargo build --release --target aarch64-linux-android
cp "$ENGINE_DIR/target/aarch64-linux-android/release/libchess_engine.so" "$JNI_DIR/arm64-v8a/"

echo "--- armeabi-v7a ---"
cd "$ENGINE_DIR" && cargo build --release --target armv7-linux-androideabi
cp "$ENGINE_DIR/target/armv7-linux-androideabi/release/libchess_engine.so" "$JNI_DIR/armeabi-v7a/"

echo "--- x86_64 ---"
cd "$ENGINE_DIR" && cargo build --release --target x86_64-linux-android
cp "$ENGINE_DIR/target/x86_64-linux-android/release/libchess_engine.so" "$JNI_DIR/x86_64/"

echo "--- x86 ---"
cd "$ENGINE_DIR" && cargo build --release --target i686-linux-android
cp "$ENGINE_DIR/target/i686-linux-android/release/libchess_engine.so" "$JNI_DIR/x86/"

echo "=== Android build complete ==="
ls -la "$JNI_DIR/arm64-v8a/" "$JNI_DIR/armeabi-v7a/" "$JNI_DIR/x86_64/" "$JNI_DIR/x86/"
