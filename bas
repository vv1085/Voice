#!/usr/bin/env bash
set -e

echo "Building VOICE artifacts..."

python scripts/build_artifacts.py

echo "Checking artifacts..."
ls -lah artifacts/

echo "Build completed successfully."
