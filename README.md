# Modal Chat – vLLM server + terminal UI

Run any vLLM-compatible model on [Modal](https://modal.com) and chat with it from
your terminal.

* **Server** (`vllm_server.py`) – spins up GPUs on demand and exposes an **OpenAI
  compatible** `/v1` HTTP endpoint.
* **Client** (`textual_client.py`) – a lightweight
  [Textual](https://textual.textualize.io/) interface that streams tokens live and shows the model's intermediate reasoning.

---

## Quick start

```bash
uv sync && source .venv/bin/activate      # install deps (Python 3.12+)
modal deploy vllm_server.py               # launch the server on Modal
export OPENAI_BASE_URL=https://<user>--vllm-server-serve.modal.run/v1
python textual_client.py                  # open the chat UI
```

Keyboard shortcuts: **Ctrl-Q** quit · **Ctrl-R** reset · **Ctrl-1/2/3** reasoning effort.

---

## Customize the server

Edit the constants at the top of `vllm_server.py` to change the model,
revision, GPU type/count or max context length.  The script mounts HF and vLLM
cache volumes so reloads are fast.

---

## Requirements

* Python 3.12+
* [uv](https://github.com/astral-sh/uv)
* A Modal account with GPU quota

---

## License

Apache 2.0 – see `LICENSE` for details.
