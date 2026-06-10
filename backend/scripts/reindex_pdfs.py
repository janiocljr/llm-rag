#!/usr/bin/env python3

import sys
from pathlib import Path

def reindex():
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from app.core.config import settings
    from app.core.ingestion import PDFIngester

    print("🔄 Re-indexing PDFs with improved table context...")
    print(f"📁 PDFs location: {settings.data_dir}")
    print(f"📁 Index location: {settings.index_dir}")
    print()

    ingester = PDFIngester(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        index_dir=settings.index_dir,
    )

    print("🗑️  Clearing old index...")
    index_file = settings.index_dir / "faiss.index"
    metadata_file = settings.index_dir / "metadata.json"

    if index_file.exists():
        index_file.unlink()
        print(f"   ✓ Deleted {index_file}")
    if metadata_file.exists():
        metadata_file.unlink()
        print(f"   ✓ Deleted {metadata_file}")

    print()
    print("📚 Loading and processing PDFs...")

    chunks = ingester.load_directory(settings.data_dir)
    chunks_list = list(chunks)

    if not chunks_list:
        print("❌ No PDFs found in data directory!")
        return False

    print(f"✅ Extracted {len(chunks_list)} chunks from PDFs")
    print()
    print("Statistics by document:")

    docs = {}
    for chunk in chunks_list:
        doc = chunk.source_file
        if doc not in docs:
            docs[doc] = {"count": 0, "tokens": 0}
        docs[doc]["count"] += 1
        docs[doc]["tokens"] += chunk.token_estimate

    for doc, stats in sorted(docs.items()):
        print(f"  • {doc}: {stats['count']} chunks, ~{stats['tokens']} tokens")

    return True


if __name__ == "__main__":
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    success = reindex()
    if success:
        print()
        print("✨ Re-indexing complete!")
        print("📌 Next step: Restart the backend to use new index")
        print("   python start.py")
    else:
        print()
        print("❌ Re-indexing failed")
        sys.exit(1)
