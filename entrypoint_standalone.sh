#!/bin/bash

# Standalone entrypoint for GPU pods (Novita, Vast.ai, etc.)
# Downloads models on first start, then runs ComfyUI + FastAPI

set -e

echo "============================================"
echo "  Qwen Image Edit - Standalone Pod Mode"
echo "============================================"

# CUDA check
echo "Checking CUDA availability..."

python_cuda_check() {
    python3 -c "
import torch
try:
    if torch.cuda.is_available():
        print('CUDA_AVAILABLE')
        exit(0)
    else:
        print('CUDA_NOT_AVAILABLE')
        exit(1)
except Exception as e:
    print(f'CUDA_ERROR: {e}')
    exit(2)
" 2>/dev/null
}

cuda_status=$(python_cuda_check)
case $? in
    0)
        echo "✅ CUDA is available and working"
        export CUDA_VISIBLE_DEVICES=0
        export FORCE_CUDA=1
        ;;
    1)
        echo "❌ CUDA is not available"
        echo "Error: CUDA is required but not available. Exiting..."
        exit 1
        ;;
    2)
        echo "❌ CUDA check failed"
        echo "Error: CUDA initialization failed. Exiting..."
        exit 1
        ;;
esac

# Additional nvidia-smi check
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA driver working"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo "❌ NVIDIA driver found but not working"
        exit 1
    fi
else
    echo "❌ NVIDIA driver not found"
    exit 1
fi

echo "Using CUDA device: $CUDA_VISIBLE_DEVICES"

# Download models if needed (first start)
echo ""
/download_models.sh

# Start ComfyUI in the background
echo ""
echo "Starting ComfyUI on port 8188..."
python /ComfyUI/main.py --listen --use-sage-attention &
COMFYUI_PID=$!

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to be ready..."
max_wait=300  # 5 minutes
wait_count=0
while [ $wait_count -lt $max_wait ]; do
    if curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; then
        echo "✅ ComfyUI is ready!"
        break
    fi
    echo "Waiting for ComfyUI... ($wait_count/$max_wait seconds)"
    sleep 2
    wait_count=$((wait_count + 2))
done

if [ $wait_count -ge $max_wait ]; then
    echo "❌ ComfyUI failed to start within $max_wait seconds"
    exit 1
fi

# Hourly cleanup. /generate returns images inline, so nothing on disk is needed once a
# request finishes; task_* input dirs leak when a request times out or errors.
# Every command must tolerate failure: set -e is inherited and would end the loop.
echo "Starting hourly cleanup (files older than 60 minutes)..."
(
    while true; do
        removed=$(find /ComfyUI/output /ComfyUI/temp -type f -mmin +60 -print -delete 2>/dev/null | wc -l) || removed=0
        find / -maxdepth 1 -type d -name 'task_*' -mmin +60 -exec rm -rf {} + 2>/dev/null || true
        echo "[cleanup] $(date '+%Y-%m-%d %H:%M:%S') removed $removed old files"
        sleep 3600
    done
) &

# Start FastAPI server
echo ""
echo "Starting FastAPI server on port 8000..."
echo "============================================"
echo "  API Endpoints:"
echo "    POST /generate  - Generate edited image"
echo "    GET  /health    - Health check"
echo "    GET  /docs      - API documentation"
echo "============================================"
echo ""

# Run uvicorn in foreground (main process)
exec python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
