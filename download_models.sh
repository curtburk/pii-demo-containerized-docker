#!/bin/bash

# Download GGUF models to the local models/ directory
# These get mounted into the container at runtime

echo ""
echo "Downloading pii_detector model (~18GB)..."
echo "Downloading tinyllama model (~700MB)..."

pip install huggingface_hub
huggingface-cli download curtburk/pii-masking-demo-models \
    --local-dir ./models \
    --local-dir-use-symlinks False


echo ""
echo "✅ Models downloaded to ./models/"
echo "   You can now run: docker compose up --build"
