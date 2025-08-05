import modal

# Configuration
MODEL_NAME = "openai/gpt-oss-120b"  # Change to any vLLM-compatible model
MODEL_REVISION = None  # Optional: specific revision
GPU_TYPE = "H100"  # Options: A100, H100, B200, etc.
N_GPU = 1
cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Set up environment
vllm_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .entrypoint([])  # remove verbose logging by base image on entry
    .uv_pip_install(
        "vllm==0.10.1+gptoss",
        extra_index_url="https://wheels.vllm.ai/gpt-oss/",
        pre=True,
        extra_options="--extra-index-url https://download.pytorch.org/whl/nightly/cu128 --index-strategy unsafe-best-match",
    )
    .uv_pip_install("huggingface_hub[hf_transfer]")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_USE_V1": "1",
        }
    )
)

# Create volumes for caching
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# Create app
app = modal.App("vllm-server")


# Serve function
@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=15 * 60,  # 15 minutes
    timeout=10 * 60,  # 10 minutes
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=10 * 60)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--tensor-parallel-size",
        str(N_GPU),
    ]

    if MODEL_REVISION:
        cmd.extend(["--revision", MODEL_REVISION])

    subprocess.Popen(" ".join(cmd), shell=True)
