#!/usr/bin/env python3

import os
import sys
from pathlib import Path


def _model_cache_exists(model_id: str) -> bool:
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    cache_name = "models--" + model_id.replace("/", "--")
    return (cache_dir / cache_name).exists()


def _read_model_from_env() -> str:
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("EMBEDDING_MODEL") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"\'')
    return "intfloat/multilingual-e5-small"


def ensure_embedding_model(model_id: str) -> bool:
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    if _model_cache_exists(model_id):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print(f"[startup] Using cached model: {model_id} (offline mode)", flush=True)
    else:
        os.environ["REQUESTS_CA_BUNDLE"] = ""
        os.environ["CURL_CA_BUNDLE"] = ""
        print(f"[startup] Downloading model: {model_id} (SSL verification bypassed)", flush=True)

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_id)
        dim = model.get_sentence_embedding_dimension()
        print(f"✅ Embedding model ready: {model_id} (dim={dim})", flush=True)
        return True
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}", flush=True)
        return False


if __name__ == "__main__":
    model = _read_model_from_env()
    print(f"[startup] Ensuring embedding model: {model}", flush=True)
    if not ensure_embedding_model(model):
        print("[startup] ⚠️  Embedding model failed to load", flush=True)
        sys.exit(1)
    print("[startup] ✅ Models ready", flush=True)
