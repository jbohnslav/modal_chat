# vLLM Server on Modal

A minimal script to run any vLLM-compatible model as an OpenAI-compatible server on Modal.

## Usage

1. Install Modal:
   ```bash
   pip install modal
   ```

2. Configure your model in `main.py`:
   - `MODEL_NAME`: Any Hugging Face model compatible with vLLM
   - `GPU_TYPE`: A100, H100, or B200
   - `N_GPU`: Number of GPUs (for tensor parallelism)

3. Deploy:
   ```bash
   modal deploy main.py
   ```

4. Use the deployed URL with any OpenAI-compatible client:
   ```python
   from openai import OpenAI
   
   client = OpenAI(
       base_url="https://your-workspace--vllm-server-serve.modal.run/v1",
       api_key="dummy"  # vLLM doesn't require auth
   )
   
   response = client.chat.completions.create(
       model="your-model-name",
       messages=[{"role": "user", "content": "Hello!"}]
   )
   ```

## Features

- Automatic model caching with Modal Volumes
- OpenAI-compatible API
- Optimized with vLLM V1 engine
- Support for FP8 quantized models
- Auto-scaling with 15-minute scaledown window