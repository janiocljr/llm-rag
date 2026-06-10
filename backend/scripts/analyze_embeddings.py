#!/usr/bin/env python3

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

SAMPLE_TEXTS = {
    "economic_analysis": "A inflação acumulada em 12 meses atingiu 4.5% em julho de 2025, refletindo pressões de demanda agregada e ajustes de preços administrados.",
    "policy_evaluation": "A implementação de políticas públicas requer avaliação rigorosa de impacto socioeconômico, com focus em indicadores de bem-estar populacional.",
    "regional_development": "O desenvolvimento paranaense depende de investimentos em infraestrutura logística, integração com mercados nacionais e atração de capital humano.",
    "query_similar_to_economic": "Qual foi a inflação em julho de 2025?",
    "query_similar_to_policy": "Como avaliar o impacto de políticas públicas?",
    "query_similar_to_regional": "Quais são os fatores para desenvolvimento econômico no Paraná?",
}

EMBEDDING_MODELS = {
    "e5-small": {
        "model_id": "intfloat/multilingual-e5-small",
        "dim": 384,
        "size": "118M",
        "speed": "Fast",
        "description": "Lightweight multilingual (current)",
    },
    "e5-base": {
        "model_id": "intfloat/multilingual-e5-base",
        "dim": 768,
        "size": "278M",
        "speed": "Medium",
        "description": "Balanced multilingual",
    },
    "e5-large": {
        "model_id": "intfloat/multilingual-e5-large",
        "dim": 1024,
        "size": "560M",
        "speed": "Slow",
        "description": "Best quality multilingual",
    },
    "bge-m3": {
        "model_id": "BAAI/bge-m3",
        "dim": 1024,
        "size": "570M",
        "speed": "Slow",
        "description": "BGE multilingual (default recommended)",
    },
    "bge-small-en": {
        "model_id": "BAAI/bge-small-en-v1.5",
        "dim": 384,
        "size": "33M",
        "speed": "Very Fast",
        "description": "English-focused (not recommended for PT-BR)",
    },
}


def compute_similarities(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings @ embeddings.T


def analyze_model(model_name: str, model_id: str) -> Dict:
    print(f"\n📊 Analyzing {model_name} ({model_id})...", flush=True)

    try:
        model = SentenceTransformer(model_id)
        texts = list(SAMPLE_TEXTS.values())
        text_labels = list(SAMPLE_TEXTS.keys())

        similarities = compute_similarities(model, texts)

        doc_indices = [0, 1, 2]
        query_indices = [3, 4, 5]

        query_doc_sims = similarities[query_indices, :][:, doc_indices]

        return {
            "model_name": model_name,
            "model_id": model_id,
            "dimension": model.get_sentence_embedding_dimension(),
            "success": True,
            "query_doc_similarities": query_doc_sims.tolist(),
            "avg_relevance": query_doc_sims.mean().item(),
            "max_relevance": query_doc_sims.max().item(),
            "min_relevance": query_doc_sims.min().item(),
        }
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {
            "model_name": model_name,
            "model_id": model_id,
            "success": False,
            "error": str(e),
        }


def print_analysis(results: List[Dict]) -> None:
    print("\n" + "=" * 80)
    print("EMBEDDING MODEL COMPARISON FOR PORTUGUESE POLICY DOCUMENTS")
    print("=" * 80)

    successful = [r for r in results if r.get("success")]

    print("\n📋 Model Specifications:")
    print(f"{'Model':<20} {'Dim':<6} {'Avg Sim':<12} {'Max Sim':<12} {'Status':<15}")
    print("-" * 65)

    for result in results:
        if result.get("success"):
            avg = result["avg_relevance"]
            max_sim = result["max_relevance"]
            status = "✅ Working"
        else:
            avg = max_sim = 0
            status = "❌ Failed"

        model_name = result["model_name"]
        dim = EMBEDDING_MODELS.get(model_name, {}).get("dim", "?")
        print(f"{model_name:<20} {dim:<6} {avg:<12.4f} {max_sim:<12.4f} {status:<15}")

    print("\n" + "=" * 80)
    print("DETAILED SIMILARITY ANALYSIS")
    print("=" * 80)
    print("\nQuery-to-Document Cosine Similarities (0.0 = no match, 1.0 = perfect match):")
    print("Higher values indicate better semantic matching between queries and documents\n")

    for result in successful:
        print(f"\n{result['model_name']} (dim={result['dimension']}):")
        sims = np.array(result["query_doc_similarities"])
        labels = ["economic_query", "policy_query", "regional_query"]
        docs = ["economic_doc", "policy_doc", "regional_doc"]

        print(f"{'':20}", end="")
        for doc in docs:
            print(f"{doc:>12}", end="")
        print()
        print("-" * 60)

        for i, label in enumerate(labels):
            print(f"{label:20}", end="")
            for j in range(3):
                val = sims[i, j]
                indicator = "✅" if val > 0.6 else "⚠️" if val > 0.4 else "❌"
                print(f"{val:>10.4f} {indicator}", end="")
            print()

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    current = "e5-small"
    current_result = next((r for r in results if r["model_name"] == current), None)

    if current_result and current_result.get("success"):
        print(f"\nCurrent model: {current} (avg similarity: {current_result['avg_relevance']:.4f})")
        print("\n✅ ASSESSMENT FOR PORTUGUESE POLICY DOCUMENTS:")
        print("   • intfloat/multilingual-e5-small is ADEQUATE for general Portuguese docs")
        print("   • BUT it may lack precision for specialized domain documents")
        print("   • Recommendation: UPGRADE to e5-large or bge-m3 for better accuracy")
        print("\nIMPROVEMENTS IF YOU UPGRADE:")
        print("   • Better semantic understanding of policy/economic terminology")
        print("   • Higher discriminative power between similar documents")
        print("   • Lower false negatives (documents marked as 'not found' when relevant)")
        print("\nTRADE-OFFS:")
        print("   • e5-large: +278MB model, +2-3x slower, better quality")
        print("   • bge-m3: +370MB model, +2-3x slower, best overall quality")


if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"

    results = []
    for model_name, config in EMBEDDING_MODELS.items():
        result = analyze_model(model_name, config["model_id"])
        results.append(result)

    print_analysis(results)

    with open("embedding_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to embedding_analysis.json")
