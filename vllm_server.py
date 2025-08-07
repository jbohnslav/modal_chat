import json
import subprocess
import time
import urllib.request

import modal

# Configuration
MODEL_NAME = "openai/gpt-oss-120b"  # Change to any vLLM-compatible model
MODEL_REVISION = None  # Optional: specific revision
GPU_TYPE = "H100"  # Options: A100, H100, B200, etc.
N_GPU = 1
MAX_MODEL_LEN = 65536
cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
MINUTES = 60

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
            "TORCH_CUDA_ARCH_LIST": "9.0;10.0",  # H100/H200 (9.0) and B200 (10.0)
        }
    )
)

# Create volumes for caching
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# Create app
app = modal.App("vllm-server")


def _wait_for_http(url: str, timeout_seconds: int = 10 * 60, poll_interval: float = 1.0) -> None:
    """Wait for an HTTP endpoint to become available (2xx/3xx/4xx considered up)."""
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:  # nosec - local call
                # If the server responds at all, we consider it up
                if 200 <= resp.status < 600:
                    return
        except Exception as e:  # noqa: BLE001
            last_error = e
            time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for {url!r}. Last error: {last_error}")


def _post_json(url: str, payload: dict, timeout_seconds: int = 30) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:  # nosec - local call
        # Read response to completion to ensure any lazy alloc/compile happens
        _ = resp.read()


@app.cls(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=15 * 60,
    timeout=10 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class VLLMServer:
    """Modal class that launches vLLM and snapshots CPU+GPU memory after warmup."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None

    @modal.enter()
    def _enter(self) -> None:
        # Launch vLLM HTTP server
        cmd: list[str] = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--tensor-parallel-size",
            str(N_GPU),
            "--max-model-len",
            str(MAX_MODEL_LEN),
        ]
        if MODEL_REVISION:
            cmd.extend(["--revision", str(MODEL_REVISION)])

        # Use exec form to avoid shell, capture the process handle
        self._proc = subprocess.Popen(cmd, text=True)

        # Wait until the server responds locally
        _wait_for_http("http://127.0.0.1:8000/v1/models", timeout_seconds=10 * 60)

        # Run a minimal request to force any lazy GPU allocations / kernel compiles
        try:
            _post_json(
                "http://127.0.0.1:8000/v1/chat/completions",
                payload={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                    "temperature": 0,
                },
                timeout_seconds=60,
            )
        except Exception:
            # Best-effort warmup; proceed to snapshot regardless
            pass

    @modal.web_server(port=8000, startup_timeout=10 * 60)
    def serve(self) -> None:
        # Keep the container alive by waiting on the vLLM process
        assert self._proc is not None
        self._proc.wait()
