"""
Standalone FastAPI server for Qwen Image Edit.
Provides the same API interface as the RunPod handler but runs as a standalone HTTP server.
"""

import os
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii
import subprocess
import time
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

import websocket
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server configuration
SERVER_ADDRESS = os.getenv('COMFYUI_ADDRESS', '127.0.0.1')
API_PORT = int(os.getenv('API_PORT', '8000'))
client_id = str(uuid.uuid4())


# --- Request/Response Models ---

class GenerateRequest(BaseModel):
    prompt: str
    # Image input options (use one)
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    # Second image input options (optional, for 2-image workflows)
    image_path_2: Optional[str] = None
    image_url_2: Optional[str] = None
    image_base64_2: Optional[str] = None
    # Generation parameters
    seed: int = 0
    steps: int = 4
    width: int = 512
    height: int = 512
    resize_mode: str = "stretch"  # stretch, crop, pad, disabled


class GenerateResponse(BaseModel):
    image: str  # Base64 encoded image
    

class HealthResponse(BaseModel):
    status: str
    comfyui: str
    cuda: str


# --- Utility Functions ---

def check_cuda_availability() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA is available and working")
            return True
        else:
            logger.warning("CUDA is not available")
            return False
    except Exception as e:
        logger.error(f"CUDA check failed: {e}")
        return False


def check_comfyui_ready() -> bool:
    """Check if ComfyUI server is ready."""
    try:
        url = f"http://{SERVER_ADDRESS}:8188/"
        urllib.request.urlopen(url, timeout=5)
        return True
    except Exception:
        return False


def download_file_from_url(url: str, output_path: str) -> str:
    """Download file from URL using wget."""
    try:
        result = subprocess.run(
            ['wget', '-O', output_path, '--no-verbose', url],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            logger.info(f"Downloaded: {url} -> {output_path}")
            return output_path
        else:
            raise Exception(f"wget failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise Exception("Download timeout")
    except Exception as e:
        raise Exception(f"Download error: {e}")


def save_base64_to_file(base64_data: str, temp_dir: str, output_filename: str) -> str:
    """Save base64 data to file."""
    try:
        decoded_data = base64.b64decode(base64_data)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
        logger.info(f"Saved base64 to: {file_path}")
        return file_path
    except (binascii.Error, ValueError) as e:
        raise Exception(f"Base64 decode failed: {e}")


def process_input(input_data: str, temp_dir: str, output_filename: str, input_type: str) -> str:
    """Process input data and return file path."""
    if input_type == "path":
        return input_data
    elif input_type == "url":
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"Unsupported input type: {input_type}")


def queue_prompt(prompt: dict) -> dict:
    """Queue a prompt to ComfyUI."""
    url = f"http://{SERVER_ADDRESS}:8188/prompt"
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_image(filename: str, subfolder: str, folder_type: str) -> bytes:
    """Get image from ComfyUI."""
    url = f"http://{SERVER_ADDRESS}:8188/view"
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}") as response:
        return response.read()


def get_history(prompt_id: str) -> dict:
    """Get history from ComfyUI."""
    url = f"http://{SERVER_ADDRESS}:8188/history/{prompt_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def get_images(ws, prompt: dict) -> dict:
    """Execute prompt and get resulting images."""
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
        else:
            continue

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                if isinstance(image_data, bytes):
                    image_data = base64.b64encode(image_data).decode('utf-8')
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images


def load_workflow(workflow_path: str) -> dict:
    """Load workflow JSON file."""
    with open(workflow_path, 'r') as file:
        return json.load(file)


# --- FastAPI Application ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting API server...")
    cuda_available = check_cuda_availability()
    if not cuda_available:
        logger.warning("CUDA not available - generation may fail")
    
    # Wait for ComfyUI to be ready
    logger.info("Waiting for ComfyUI to be ready...")
    max_wait = 300  # 5 minutes
    waited = 0
    while waited < max_wait:
        if check_comfyui_ready():
            logger.info("ComfyUI is ready!")
            break
        await asyncio.sleep(2)
        waited += 2
    
    if waited >= max_wait:
        logger.error("ComfyUI failed to start within timeout")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")


app = FastAPI(
    title="Qwen Image Edit API",
    description="Standalone API for Qwen Image Edit using ComfyUI backend",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    cuda_ok = check_cuda_availability()
    comfyui_ok = check_comfyui_ready()
    
    return HealthResponse(
        status="healthy" if (cuda_ok and comfyui_ok) else "degraded",
        comfyui="ready" if comfyui_ok else "not_ready",
        cuda="available" if cuda_ok else "not_available"
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate edited image."""
    task_id = f"task_{uuid.uuid4()}"
    
    # Process first image input
    image1_path = None
    if request.image_path:
        image1_path = process_input(request.image_path, task_id, "input_image_1.jpg", "path")
    elif request.image_url:
        image1_path = process_input(request.image_url, task_id, "input_image_1.jpg", "url")
    elif request.image_base64:
        image1_path = process_input(request.image_base64, task_id, "input_image_1.jpg", "base64")
    else:
        raise HTTPException(status_code=400, detail="No image input provided. Use image_path, image_url, or image_base64")
    
    # Process second image input (optional)
    image2_path = None
    if request.image_path_2:
        image2_path = process_input(request.image_path_2, task_id, "input_image_2.jpg", "path")
    elif request.image_url_2:
        image2_path = process_input(request.image_url_2, task_id, "input_image_2.jpg", "url")
    elif request.image_base64_2:
        image2_path = process_input(request.image_base64_2, task_id, "input_image_2.jpg", "base64")
    
    # Select workflow based on number of images
    if image2_path:
        workflow_path = "/qwen_image_edit_2.json"
    else:
        workflow_path = "/qwen_image_edit_1.json"
    
    prompt = load_workflow(workflow_path)
    
    # Set image inputs
    prompt["78"]["inputs"]["image"] = image1_path
    if image2_path:
        prompt["123"]["inputs"]["image"] = image2_path
    
    # Set prompt text
    prompt["111"]["inputs"]["prompt"] = request.prompt
    
    # Set generation parameters
    prompt["3"]["inputs"]["seed"] = request.seed
    prompt["3"]["inputs"]["steps"] = request.steps
    
    # Handle resize mode
    width = request.width
    height = request.height
    resize_mode = request.resize_mode
    
    skip_resize = (width == 0 and height == 0) or resize_mode == "disabled"
    
    if skip_resize:
        # Get original image dimensions
        with Image.open(image1_path) as img:
            width, height = img.size
        # Ensure divisible by 2 (VAE requirement)
        width = width - (width % 2)
        height = height - (height % 2)
        
        # Bypass resize node
        prompt["110"]["inputs"]["image1"] = ["78", 0]
        prompt["111"]["inputs"]["image1"] = ["78", 0]
        
        if image2_path:
            prompt["110"]["inputs"]["image2"] = ["123", 0]
            prompt["111"]["inputs"]["image2"] = ["123", 0]
    else:
        # Set resize mode
        prompt["127"]["inputs"]["keep_proportion"] = resize_mode
        if image2_path:
            prompt["130"]["inputs"]["keep_proportion"] = resize_mode
    
    # Set dimensions
    prompt["128"]["inputs"]["value"] = width
    prompt["129"]["inputs"]["value"] = height
    
    # Connect to ComfyUI WebSocket
    ws_url = f"ws://{SERVER_ADDRESS}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting to ComfyUI WebSocket: {ws_url}")
    
    try:
        ws = websocket.WebSocket()
        ws.settimeout(300)  # 5 minute timeout for generation
        ws.connect(ws_url)
        
        images = get_images(ws, prompt)
        ws.close()
        
        # Clean up temp files
        try:
            import shutil
            if os.path.exists(task_id):
                shutil.rmtree(task_id)
        except Exception as e:
            logger.warning(f"Failed to clean up temp files: {e}")
        
        # Return first image
        if not images:
            raise HTTPException(status_code=500, detail="No images generated")
        
        for node_id in images:
            if images[node_id]:
                return GenerateResponse(image=images[node_id][0])
        
        raise HTTPException(status_code=500, detail="No images found in output")
        
    except websocket.WebSocketException as e:
        raise HTTPException(status_code=503, detail=f"ComfyUI connection failed: {e}")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Qwen Image Edit API",
        "version": "1.0.0",
        "endpoints": {
            "/generate": "POST - Generate edited image",
            "/health": "GET - Health check",
            "/docs": "GET - OpenAPI documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
