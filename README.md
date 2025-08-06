# vLLM Server on Modal

```bash
uv sync                      # install dependencies
source .venv/bin/activate    # activate virtual environment
modal deploy main.py         # deploy vLLM server to Modal
export OPENAI_BASE_URL=https://{modal_username}--vllm-server-serve.modal.run/v1
curl $OPENAI_BASE_URL/models # wait for server to be ready
python textual_client.py     # run chat client
```