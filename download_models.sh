#!/bin/bash

# Download GGUF models to the local models/ directory
# These get mounted into the container at runtime

#!/bin/bash

# Download GGUF models to the local models/ directory
# These get mounted into the container at runtime

echo "================================================"
echo "PII Models Download Script"
echo "================================================"
echo ""
echo "TinyLlama (~2GB)..."
echo "Finetuned PII Detector (~20GB)..."

cd ~/Desktop/pii-demo-containerized-docker

echo "creating local models folder"
mkdir models

python3 -m venv 'new-ft-env'
source new-ft-env/bin/activate

pip install huggingface_hub

python3 << EOF
from huggingface_hub import snapshot_download
snapshot_download(repo_id="curtburk/pii-masking-demo-models", local_dir="./models")
EOF


deactivate 

echo ""
echo "✅ Models downloaded to ./models/"
echo "   You can now run: ./start.sh"


