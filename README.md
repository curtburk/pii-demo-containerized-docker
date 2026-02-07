# PII Masking Demo — Dockerized

## Overview

Containerized version of the PII Masking fine-tuning comparison demo for the HP ZGX Nano. Compares a base TinyLlama model against a fine-tuned Qwen2.5-32B model to demonstrate the value of local AI model fine-tuning.

## Prerequisites

- **HP ZGX Nano** (or any system with NVIDIA GPU + 40GB+ VRAM)
- **Docker** with Docker Compose v2
- **NVIDIA Container Toolkit** ([install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

### First-time Docker setup

Make sure the Docker daemon is running:
```bash
sudo systemctl start docker
```

To have Docker start automatically on boot:
```bash
sudo systemctl enable docker
```

Add your user to the `docker` group so you don't need `sudo` for every command:
```bash
sudo usermod -aG docker $USER
newgrp docker   # applies immediately in current shell
```

Verify your GPU is accessible to Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

## Quick Start

```bash
# 1. Download model files (~19GB total)
chmod +x download_models.sh
./download_models.sh

### these will download from a HuggingFace repo @ https://huggingface.co/curtburk/pii-masking-demo-models

# 2. Build and run
chmod +x start.sh
./start.sh --build
```

On startup, the terminal will display clickable URLs with the host's IP address:

```
============================================================
🚀 Demo is ready!
============================================================
  Frontend:  http://<your host device IP address>:8000/app/
  API:       http://<your host device IP address>:8000/
  API Docs:  http://<your host device IP address>:8000/docs
============================================================
```

For subsequent runs (no rebuild needed):
```bash
./start.sh
```

To stop:
```bash
docker compose down
```

## Directory Structure

```
pii-demo-docker/
├── start.sh                      # Launch script (auto-detects host IP)
├── Dockerfile                    # Multi-stage build (CUDA + Python)
├── docker-compose.yml            # GPU passthrough and environment config
├── download_models.sh            # Downloads GGUF models
├── backend/
│   ├── main.py                   # FastAPI server (serves API + frontend)
│   ├── requirements-docker.txt   # Slim Python dependencies
│   └── offline_responses.json    # Fallback responses
├── frontend/
│   └── index.html                # Web interface (you provide this)
└── models/                       # GGUF model files (git-ignored, mounted at runtime)
    ├── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
    └── pii_detector_Q4_K_M.gguf
```

## Architecture Changes from Bare-Metal Version

| Aspect | Bare-Metal | Docker |
|--------|-----------|--------|
| Model paths | Hardcoded absolute paths | Environment variables |
| Frontend server | Separate `python3 -m http.server` on port 8080 | Served by FastAPI on port 8000 at `/app/` |
| IP configuration | Manual edit of `start_demo_remote.sh` | Auto-detected by `start.sh` |
| Startup | `./start_demo_remote.sh` | `./start.sh` |
| Dependencies | venv + manual pip installs | Baked into image |
| Models | Stored in project directory | Mounted as Docker volume |

## Configuration

All settings are configurable via environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_MODEL_PATH` | `/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | Path to base model |
| `FINETUNED_MODEL_PATH` | `/app/models/pii_detector_Q4_K_M.gguf` | Path to fine-tuned model |
| `N_GPU_LAYERS` | `-1` (all) | Number of layers to offload to GPU |
| `N_CTX` | `2048` | Context window size |
| `N_BATCH` | `4096` | Batch size for prompt processing |
| `N_THREADS` | `8` | CPU threads for non-GPU work |
| `AUTO_LOAD_MODELS` | `true` | Load models on startup vs. manual trigger |
| `HOST_IP` | Auto-detected | Override the displayed IP address |

You can also override `HOST_IP` at launch time:
```bash
HOST_IP=10.0.0.5 ./start.sh --build
```

## Targeting Different GPUs

The Dockerfile defaults to Blackwell architecture (compute capability 120). To target a different GPU, edit the `CMAKE_CUDA_ARCHITECTURES` value in the Dockerfile:

| GPU | Architecture | Value |
|-----|-------------|-------|
| GB10 / GB200 (Blackwell) | sm_120 | `120` |
| RTX 4090 / L40 (Ada Lovelace) | sm_89 | `89` |
| A100 (Ampere) | sm_80 | `80` |

## Troubleshooting

**"Models not found" error**: Verify the `models/` directory contains both `.gguf` files and that the volume mount in `docker-compose.yml` points to the correct location.

**GPU not detected in container**: Run `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi` to verify the NVIDIA Container Toolkit is working.

**Build fails on llama-cpp-python**: This usually means the CUDA architecture doesn't match your GPU. Update `CMAKE_CUDA_ARCHITECTURES` in the Dockerfile.

**Port already in use**: `docker compose down` first, or change the port mapping in `docker-compose.yml` (e.g., `"9000:8000"`).

**Wrong IP in startup URLs**: Override with `HOST_IP=your.ip.here ./start.sh`

## Support

Curtis Burkhalter — curtis.burkhalter@hp.com
