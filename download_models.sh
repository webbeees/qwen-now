#!/bin/bash

# Download models if they don't exist
# This script is run at container startup

set -e

echo "============================================"
echo "  Checking/Downloading Models"
echo "============================================"

# Model URLs
DIFFUSION_URL="https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"
LORA_URL="https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0.safetensors"
TEXT_ENCODER_URL="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE_URL="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"

# Model paths
DIFFUSION_PATH="/ComfyUI/models/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"
LORA_PATH="/ComfyUI/models/loras/Qwen-Image-Lightning-4steps-V1.0.safetensors"
TEXT_ENCODER_PATH="/ComfyUI/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE_PATH="/ComfyUI/models/vae/qwen_image_vae.safetensors"

download_if_missing() {
    local url="$1"
    local path="$2"
    local name="$3"
    
    if [ -f "$path" ]; then
        echo "✅ $name already exists ($(du -h "$path" | cut -f1))"
    else
        echo "⬇️  Downloading $name..."
        wget --progress=bar:force -O "$path" "$url"
        echo "✅ $name downloaded ($(du -h "$path" | cut -f1))"
    fi
}

# Download models in parallel for speed
echo ""
echo "Starting parallel downloads..."

download_if_missing "$DIFFUSION_URL" "$DIFFUSION_PATH" "Diffusion Model" &
PID1=$!

download_if_missing "$LORA_URL" "$LORA_PATH" "LoRA Model" &
PID2=$!

download_if_missing "$TEXT_ENCODER_URL" "$TEXT_ENCODER_PATH" "Text Encoder" &
PID3=$!

download_if_missing "$VAE_URL" "$VAE_PATH" "VAE" &
PID4=$!

# Wait for all downloads
echo "Waiting for downloads to complete..."
wait $PID1 $PID2 $PID3 $PID4

echo ""
echo "============================================"
echo "  All models ready!"
echo "============================================"
echo ""

# Verify all models exist
if [ ! -f "$DIFFUSION_PATH" ] || [ ! -f "$LORA_PATH" ] || [ ! -f "$TEXT_ENCODER_PATH" ] || [ ! -f "$VAE_PATH" ]; then
    echo "❌ ERROR: Some models are missing!"
    exit 1
fi

echo "Model sizes:"
du -h "$DIFFUSION_PATH" "$LORA_PATH" "$TEXT_ENCODER_PATH" "$VAE_PATH"
