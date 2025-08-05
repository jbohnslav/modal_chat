import json
import os

from openai import OpenAI

client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key="EMPTY")

response = client.responses.create(
    model="openai/gpt-oss-120b",
    instructions="You are a helpful assistant.",
    input="Explain what MXFP4 quantization is.",
)

with open("response.json", "w") as f:
    json.dump(response.model_dump(), f, indent=4)
