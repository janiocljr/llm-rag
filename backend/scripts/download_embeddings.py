#!/usr/bin/env python3
import os
import sys
from pathlib import Path

EMBEDDING_MODELS = {
    "bge-m3": {
        "model_id": "BAAI/bge-m3",
        "description": "BGE Multilingual v3 (570 M, 1024-dim, 100+ languages) [DEFAULT]",
    },
    "multilingual-e5-small": {
        "model_id": "intfloat/multilingual-e5-small",
        "description": "Multilingual E5 Small (118 M, 384-dim, lightweight)",
    },
    "multilingual-e5-base": {
        "model_id": "intfloat/multilingual-e5-base",
        "description": "Multilingual E5 Base (278 M, 768-dim, balanced)",
    },
}

DEFAULT_MODEL = "multilingual-e5-small"


def is_model_cached() -> bool:
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_dir.exists():
        return False
    return len(list(cache_dir.glob("*"))) > 0


def download_embedding_model(model_key: str = DEFAULT_MODEL, quiet: bool = False) -> bool:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        if not quiet:
            print("❌ sentence-transformers not installed")
            print("   Run: pip install sentence-transformers")
        return False

    if model_key not in EMBEDDING_MODELS:
        if not quiet:
            print(f"❌ Unknown model: {model_key}")
            print(f"   Available: {', '.join(EMBEDDING_MODELS.keys())}")
        return False

    model_info = EMBEDDING_MODELS[model_key]
    model_id = model_info["model_id"]

    if not quiet:
        print(f"⬇   Downloading embedding model: {model_info['description']}")
        print(f"    Model ID: {model_id}")
        print("    Cache location: ~/.cache/huggingface/hub/")
        print()

    try:
        model = SentenceTransformer(model_id)
        dim = model.get_sentence_embedding_dimension()
        if not quiet:
            print("✅  Model cached successfully!")
            print(f"    Embedding dimension: {dim}")
            print()
            print("You can now deploy with confidence — the model is cached locally.")
        return True
    except Exception as e:
        if not quiet:
            print(f"❌  Download failed: {e}")
            print()
            print("Troubleshooting:")
            print("  1. Check internet connection")
            print("  2. Try disabling SSL verification (development only):")
            print("     export HF_HUB_DISABLE_TELEMETRY=1")
            print("     export REQUESTS_CA_BUNDLE=''")
            print("     python scripts/download_embeddings.py")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-download embedding models to avoid SSL issues during deployment"
    )
    parser.add_argument(
        "--model",
        choices=list(EMBEDDING_MODELS.keys()),
        default=DEFAULT_MODEL,
        help=f"Which embedding model to download (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output (useful for startup scripts)",
    )
    args = parser.parse_args()

    if args.list:
        print("Available embedding models:")
        for key, info in EMBEDDING_MODELS.items():
            default_tag = " [DEFAULT]" if key == DEFAULT_MODEL else ""
            print(f"  {key:<25} {info['description']}{default_tag}")
        sys.exit(0)

    if args.all:
        success = True
        for model_key in EMBEDDING_MODELS.keys():
            if not download_embedding_model(model_key, quiet=args.quiet):
                success = False
            if not args.quiet:
                print()
        sys.exit(0 if success else 1)
    else:
        success = download_embedding_model(args.model, quiet=args.quiet)
        if not args.quiet:
            print("Next steps:")
            print("  1. Start the backend: python -m uvicorn app.main:app --reload")
            print("  2. Or for deployment, use the cached models with confidence")
        sys.exit(0 if success else 1)
