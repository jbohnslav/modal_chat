# vLLM Server on Modal

```bash
uv sync                      # install dependencies
source .venv/bin/activate    # activate virtual environment
modal deploy main.py         # deploy vLLM server to Modal
python textual_client.py     # run chat client
```