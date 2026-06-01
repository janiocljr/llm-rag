# Guia de Avaliação e Métricas

## 1. Métricas de Recuperação (Retrieval)

### 1.1 Score de Similaridade Coseno
- **Faixa:** 0.0 – 1.0
- **Interpretação:**
  - ≥ 0.75 → alta relevância
  - 0.50 – 0.74 → relevância moderada
  - 0.35 – 0.49 → relevância baixa (limiar padrão)
  - < 0.35 → possivelmente irrelevante

### 1.2 Hit Rate @ K
Proporção de queries em que pelo menos 1 chunk relevante está entre os top-K recuperados.

```python
hit_rate = sum(1 for q in queries if any_relevant_in_top_k(q)) / len(queries)
```

### 1.3 MRR (Mean Reciprocal Rank)
Avalia a posição do primeiro chunk relevante recuperado.

---

## 2. Métricas de Geração (LLM)

### 2.1 Faithfulness
Proporção de afirmações na resposta que podem ser verificadas no contexto recuperado.  
Verificação manual ou via LLM-as-judge.

### 2.2 Answer Relevance
O quanto a resposta efetivamente responde à pergunta feita.

### 2.3 Latência
- Alvo: < 15s em CPU i7 8-core
- Breakdown: embedding (< 100ms) + FAISS search (< 10ms) + LLM (bulk do tempo)

---

## 3. Protocolo de Avaliação Manual

### 3.1 Conjunto de avaliação sugerido (por domínio)

Para cada PDF indexado, criar 5–10 perguntas com resposta esperada conhecida:

| # | Pergunta | Resposta esperada | Chunk esperado |
|---|----------|-------------------|----------------|
| 1 | Qual o crescimento do PIB no cenário tendencial? | 2,0% ao ano | doc_p2_c0 |
| 2 | Qual a taxa de inadimplência em junho/2025? | 3,5% | doc_p5_c1 |

### 3.2 Critérios de avaliação

Para cada resposta, avaliar:
- **[C]** Correto — informação presente e correta
- **[P]** Parcial — informação incompleta mas não errada
- **[I]** Incorreto — informação errada ou alucinada
- **[NF]** Não encontrado — sistema retornou `found_in_documents: false`

---

## 4. Benchmarks de Referência

| Configuração | Hit Rate @5 | Latência média |
|---|---|---|
| BGE-small + FlatIP | ~0.82 | ~8s |
| TF-IDF+LSA + FlatIP | ~0.61 | ~5s |
| BGE-small + HNSW | ~0.80 | ~3s |

> Valores estimados para corpus de ~1000 chunks. Resultados variam com domínio.

---

## 5. Rodando Avaliação Automática

```bash
# Requer conjunto de avaliação em tests/eval_dataset.json
pytest tests/ -v -m eval

# Gerar relatório de cobertura
pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html
```
