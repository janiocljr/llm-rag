"""Offline demo fixtures — used when demo_mode=True."""
from __future__ import annotations

DEMO_STATS: dict = {
    "total_chunks": 847,
    "documents": [
        "desenvolvimento_paranaense.pdf",
        "analise_economica_2025.pdf",
        "indicadores_sociais.pdf",
    ],
    "index_type": "FlatIP (exact cosine)",
    "embedding_model": "BAAI/bge-m3",
    "embedding_dim": 1024,
}

DEMO_QUERY_RESPONSE: dict = {
    "question": "Quais são os cenários de crescimento do PIB do Paraná?",
    "answer": (
        "Segundo [desenvolvimento_paranaense.pdf, p. 2], existem três cenários projetados:\n\n"
        "- **Cenário Tendencial:** Poupança 17%, Produtividade 0,5% → PIB **2,0% a.a.**\n"
        "- **Cenário Moderado:** Poupança 20%, Produtividade 1,0% → PIB **3,5% a.a.**\n"
        "- **Cenário Otimista:** Poupança 24%, Produtividade 2,0% → PIB **5,2% a.a.**\n\n"
        "O cenário otimista depende de reformas estruturais e aumento expressivo da taxa de poupança nacional."
    ),
    "retrieved_chunks": [
        {
            "chunk_id": "desenvolvimento_paranaense_p2_c0",
            "source_file": "desenvolvimento_paranaense.pdf",
            "page_number": 2,
            "chunk_index": 0,
            "score": 0.853,
            "text": (
                "Cenário Tendencial: Poupança 17%, Produtividade 0,5% → PIB 2,0% ao ano. "
                "Cenário Moderado: Poupança 20%, Produtividade 1,0% → PIB 3,5% ao ano. "
                "Cenário Otimista: Poupança 24%, Produtividade 2,0% → PIB 5,2% ao ano."
            ),
            "citation": "[desenvolvimento_paranaense.pdf, p. 2]",
        },
        {
            "chunk_id": "analise_economica_2025_p5_c1",
            "source_file": "analise_economica_2025.pdf",
            "page_number": 5,
            "chunk_index": 1,
            "score": 0.721,
            "text": (
                "A taxa de poupança nacional é apontada como principal gargalo para o crescimento sustentado "
                "do produto interno bruto nos estados da região Sul do Brasil."
            ),
            "citation": "[analise_economica_2025.pdf, p. 5]",
        },
        {
            "chunk_id": "indicadores_sociais_p8_c0",
            "source_file": "indicadores_sociais.pdf",
            "page_number": 8,
            "chunk_index": 0,
            "score": 0.594,
            "text": (
                "Reformas estruturais em infraestrutura logística e educação são necessárias para atingir "
                "o patamar de produtividade exigido pelo cenário otimista de crescimento."
            ),
            "citation": "[indicadores_sociais.pdf, p. 8]",
        },
    ],
    "full_prompt": (
        "[INST] <<SYS>>\n"
        "Você é um assistente especializado em análise de documentos. "
        "Responda APENAS com base no contexto fornecido.\n"
        "<</SYS>>\n\n"
        "<contexto>\n"
        "[desenvolvimento_paranaense.pdf, p. 2]\n"
        "Cenário Tendencial: Poupança 17%, Produtividade 0,5% → PIB 2,0% ao ano...\n"
        "</contexto>\n\n"
        "Pergunta: Quais são os cenários de crescimento do PIB do Paraná? [/INST]"
    ),
    "found_in_documents": True,
    "latency_ms": 8340.0,
}
