#!/usr/bin/env python3

import os
import sys
import urllib.request
from pathlib import Path


MODEL_DIR = Path("models")

MODELS = {
    "mistral-7b": {
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "url": (
            "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/"
            "resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        ),
        "size_gb": 4.1,
        "description": "Mistral 7B Instruct v0.2 — Q4_K_M quantised (~4.1 GB)",
    },
    "llama3-8b": {
        "filename": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "url": (
            "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/"
            "resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
        ),
        "size_gb": 4.9,
        "description": "Llama 3 8B Instruct — Q4_K_M quantised (~4.9 GB)",
    },
    "gemma2-9b": {
        "filename": "gemma-2-9b-it-Q4_K_M.gguf",
        "url": (
            "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/"
            "resolve/main/gemma-2-9b-it-Q4_K_M.gguf"
        ),
        "size_gb": 5.5,
        "description": "Gemma 2 9B Instruct — Q4_K_M quantised (~5.5 GB)",
    },
        "qwen2.5-7b": {
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "url": (
            "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/"
            "resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        ),
        "size_gb": 4.7,
        "description": "Qwen2.5 7B Instruct — Q4_K_M quantised (~4.7 GB) [PADRÃO]",
    },
}

DEFAULT_MODEL = "qwen2.5-7b"


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / (1024 ** 2)
        total_mb = total_size / (1024 ** 2)
        print(f"\r {pct:3d}%  {mb:.0f}/{total_mb:.0f} MB", end="", flush=True)
    else:
        mb = downloaded / (1024 ** 2)
        print(f"\r  Downloaded: {mb:.1f} MB", end="", flush=True)


def download_model(model_key: str = DEFAULT_MODEL) -> Path:
    model_info = MODELS[model_key]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODEL_DIR / model_info["filename"]

    if dest.exists():
        print(f"✅  Model already exists: {dest}")
        return dest

    print(f"⬇   Downloading: {model_info['description']}")
    print(f"    URL: {model_info['url']}")
    print(f"    Destination: {dest}")
    print(f"    Expected size: ~{model_info['size_gb']} GB")
    print()

    try:
        urllib.request.urlretrieve(model_info["url"], dest, _progress_hook)
        print(f"\n✅  Downloaded to {dest}")
    except Exception as e:
        if dest.exists():
            dest.unlink()
        print(f"\n❌  Download failed: {e}")
        raise

    return dest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download a local LLM for the RAG system")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=DEFAULT_MODEL,
        help=f"Which model to download (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for key, info in MODELS.items():
            default_tag = " [DEFAULT]" if key == DEFAULT_MODEL else ""
            print(f"  {key:<15} {info['description']}{default_tag}")
        sys.exit(0)

    download_model(args.model)
    print("\nNext step: run the ingestion pipeline:")
    print("  python -m uvicorn app.main:app --reload")
    print("  curl -X POST http://localhost:8000/api/v1/ingest -H 'Content-Type: application/json' -d '{}'")
