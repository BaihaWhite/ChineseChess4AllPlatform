#!/bin/bash
# Build Rust chess engine for desktop (host platform)
# Requires: Rust toolchain, JDK 17+

export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-21-temurin}"
echo "Building chess-engine for desktop (host target)..."
echo "JAVA_HOME=$JAVA_HOME"

cargo build --release

if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful!"
    echo "Output: target/release/libchess_engine.so"
    ls -lh target/release/libchess_engine.so
else
    echo "Build failed!"
    exit 1
fi
