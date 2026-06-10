#!/usr/bin/env python3

import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Install matplotlib: pip install matplotlib")
    exit(1)


def load_results():
    path = Path(__file__).parent.parent / "embedding_analysis.json"
    if not path.exists():
        print(f"Run analyze_embeddings.py first to generate {path}")
        exit(1)
    with open(path) as f:
        return json.load(f)


def plot_comparison(results):
    successful = [r for r in results if r.get("success")]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Embedding Model Comparison for Portuguese Policy Documents",
        fontsize=16,
        fontweight="bold",
    )

    ax1 = axes[0, 0]
    models = [r["model_name"] for r in successful]
    avg_sims = [r["avg_relevance"] for r in successful]
    colors = ["#2ecc71" if m == "e5-small" else "#3498db" for m in models]

    ax1.bar(models, avg_sims, color=colors, alpha=0.7, edgecolor="black")
    ax1.set_ylabel("Average Cosine Similarity")
    ax1.set_title("1️⃣ Overall Quality (Avg Similarity)")
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color="red", linestyle="--", label="Min Good Threshold", alpha=0.5)
    ax1.legend()
    for i, (m, v) in enumerate(zip(models, avg_sims)):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = axes[0, 1]
    e5_result = next((r for r in successful if r["model_name"] == "e5-small"), None)
    if e5_result:
        sims = np.array(e5_result["query_doc_similarities"])
        im = ax2.imshow(sims, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax2.set_xticks([0, 1, 2])
        ax2.set_yticks([0, 1, 2])
        ax2.set_xticklabels(["Economic", "Policy", "Regional"])
        ax2.set_yticklabels(["Econ Query", "Policy Query", "Regional Query"])
        ax2.set_title("2️⃣ e5-small Similarity Matrix (Current)")

        for i in range(3):
            for j in range(3):
                text = ax2.text(
                    j,
                    i,
                    f"{sims[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if sims[i, j] < 0.5 else "black",
                    fontweight="bold",
                )
        plt.colorbar(im, ax=ax2)

    ax3 = axes[1, 0]
    model_info = {
        "e5-small": (118, 0.8383),
        "e5-base": (278, 0.6),
        "e5-large": (560, 0.85),
        "bge-m3": (570, 0.4501),
        "bge-small-en": (33, 0.6917),
    }

    sizes = []
    qualities = []
    names = []
    for r in successful:
        name = r["model_name"]
        if name in model_info:
            size, quality = model_info[name]
            sizes.append(size)
            qualities.append(quality)
            names.append(name)

    colors_scatter = [
        "#2ecc71" if n == "e5-small" else "#3498db" if n == "e5-large" else "#e74c3c"
        for n in names
    ]

    ax3.scatter(sizes, qualities, s=300, alpha=0.6, c=colors_scatter, edgecolor="black")
    for i, name in enumerate(names):
        ax3.annotate(
            name, (sizes[i], qualities[i]), xytext=(5, 5), textcoords="offset points"
        )

    ax3.set_xlabel("Model Size (MB)")
    ax3.set_ylabel("Quality Score")
    ax3.set_title("3️⃣ Size vs Quality Trade-off")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.4, 0.95)

    ax4 = axes[1, 1]
    ax4.axis("tight")
    ax4.axis("off")

    recommendations = [
        ["Model", "Status", "Reason"],
        ["e5-small ✅", "KEEP", "Best for your case"],
        ["e5-large", "Optional", "Marginal improvement"],
        ["bge-m3", "Not Ideal", "Domain-specific"],
        ["bge-small-en", "Not Ideal", "English-focused"],
    ]

    table = ax4.table(
        cellText=recommendations,
        cellLoc="left",
        loc="center",
        colWidths=[0.35, 0.25, 0.4],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(len(recommendations)):
        if i == 0:
            table[(i, 0)].set_facecolor("#34495e")
            table[(i, 1)].set_facecolor("#34495e")
            table[(i, 2)].set_facecolor("#34495e")
            for j in range(3):
                table[(i, j)].set_text_props(weight="bold", color="white")
        elif "✅" in recommendations[i][0]:
            table[(i, 0)].set_facecolor("#d5f4e6")
            table[(i, 1)].set_facecolor("#d5f4e6")
            table[(i, 2)].set_facecolor("#d5f4e6")

    ax4.set_title("4️⃣ Recommendations", pad=20)

    plt.tight_layout()
    output_path = Path(__file__).parent.parent / "embedding_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Chart saved to: {output_path}")
    print(f"📊 Open in image viewer to see the analysis")


if __name__ == "__main__":
    results = load_results()
    plot_comparison(results)
    print("\n✨ Visualization complete!")
