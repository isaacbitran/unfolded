"""Download GPT-2 small weights and tokenizer files into ./weights (needed for tests and the demo build)."""
from huggingface_hub import hf_hub_download

for f in ["model.safetensors", "vocab.json", "merges.txt", "config.json"]:
    print(hf_hub_download("openai-community/gpt2", f, local_dir="weights"))
