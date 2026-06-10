# Perguntas de Teste para RAG System

## 📋 Estratégia de Teste

Para validar se o sistema RAG está funcionando bem, você precisa testar:
1. **Recuperação Simples** - perguntas diretas com resposta única
2. **Recuperação Complexa** - perguntas que precisam integrar múltiplos contextos
3. **Edge Cases** - perguntas difíceis, ambíguas ou negativas
4. **Cobertura de Domínios** - diferentes tópicos dos PDFs

---

## ✅ TESTE 1: Recuperação Simples (Fatos Diretos)

Essas perguntas têm respostas específicas dentro dos PDFs.

### 1.1 Análise Econômica
```
P: "Qual foi a inflação acumulada em 12 meses em julho de 2025?"
E: Deve retornar valor específico (ex: 4.5%)
⚡ Teste: Se retorna "não encontrei", o threshold está muito alto
```

### 1.2 Políticas Públicas
```
P: "Qual é o objetivo principal da avaliação de políticas públicas?"
E: Deve mencionar impacto socioeconômico / bem-estar populacional
⚡ Teste: Valida se consegue extrair conceitos gerais
```

### 1.3 Desenvolvimento Regional
```
P: "Quais são os principais setores econômicos do Paraná?"
E: Deve listar setores (agricultura, indústria, etc.)
⚡ Teste: Recuperação simples de lista
```

### 1.4 Dados Específicos
```
P: "Qual foi o PIB do Paraná em 2024?"
E: Deve retornar número específico se existir no PDF
⚡ Teste: Valida precisão de números
```

---

## 🔗 TESTE 2: Recuperação Complexa (Integração de Múltiplos Chunks)

Essas perguntas precisam que o LLM combine informações de 2-3 chunks.

### 2.1 Correlações Econômicas
```
P: "Como a inflação afeta os investimentos em desenvolvimento regional?"
E: Deve conectar análise econômica + políticas de desenvolvimento
⚡ Teste: Se retorna 4-5 chunks, o sistema está bom
         Se retorna só 1-2, há problema de recuperação
```

### 2.2 Impacto de Políticas
```
P: "Quais políticas públicas são recomendadas para o desenvolvimento do Paraná?"
E: Deve integrar avaliação de políticas + contexto regional
⚡ Teste: Valida se consegue combinar conceitos de 2 documentos
```

### 2.3 Análise Crítica
```
P: "Quais são os desafios econômicos para implementação de políticas no Paraná?"
E: Deve mencionar: inflação (econ) + capacidade institucional (políticas) + fatores regionais
⚡ Teste: Alto nível de integração = sistema muito bom
```

### 2.4 Síntese de Informações
```
P: "Descreva a situação econômica e de políticas públicas do Brasil em julho 2025"
E: Deve sintetizar análise conjuntural + recomendações de políticas
⚡ Teste: Resposta extensa = 4-5 chunks recuperados corretamente
```

---

## ❌ TESTE 3: Edge Cases (Perguntas Difíceis)

Essas perguntas testam a robustez do sistema.

### 3.1 Pergunta Não Existe
```
P: "Qual é a população exata de Curitiba em 2025?"
E: Deve retornar "Não encontrei essa informação" (VERDADEIRO "não encontrei")
⚡ Teste: Sistema reconhece quando realmente não sabe (vs falso "não encontrei")
```

### 3.2 Pergunta Ambígua
```
P: "Fale sobre desenvolvimento"
E: Deve retornar contexto de desenvolvimento paranaense + políticas
⚡ Teste: Lida bem com ambiguidade
```

### 3.3 Pergunta com Contexto Implícito
```
P: "Como melhorar isso?"
E: Deve tentar interpretar o "isso" do documento
⚡ Teste: Robustez contra perguntas vagas
```

### 3.4 Pergunta Contraditória
```
P: "Quais são os efeitos negativos da inflação controlada?"
E: Pode retornar "ambíguo" ou contexto sobre trade-offs
⚡ Teste: Valida se consegue lidar com contradições
```

---

## 🎯 TESTE 4: Cobertura de Tópicos

Teste cada documento sistematicamente.

### 4.1 Análise Conjuntural (PDF 1)
```
P: "Quais foram os principais indicadores econômicos em julho de 2025?"
→ Deve retornar múltiplos indicadores (inflação, PIB, emprego, etc.)
```

### 4.2 Políticas Públicas (PDF 2)
```
P: "Como avaliar se uma política pública é efetiva?"
→ Deve mencionar metodologia de avaliação
```

### 4.3 Desenvolvimento Regional (PDF 3)
```
P: "Qual é a estratégia de desenvolvimento para o Paraná?"
→ Deve mencionar fatores específicos do estado
```

---

## 🔬 TESTE 5: Validação da Qualidade

Use essas métricas para avaliar cada resposta.

### 5.1 Verificação Visual
```
✅ BOM:
   - Retorna 3-5 chunks relevantes
   - Chunks têm score > 0.55
   - Resposta cita fontes
   - Linguagem clara em PT-BR

❌ PROBLEMA:
   - Retorna "não encontrei" mas deveria ter resposta
   - Chunks têm score < 0.45
   - Resposta vaga ou genérica
   - Mistura português + inglês
```

### 5.2 Checklist de Validação
Para cada resposta, verifique:
```
□ Respondeu a pergunta? (Sim/Não/Parcial)
□ Citou fontes? (Sim/Não)
□ Score de chunks > 0.55? (Sim/Não)
□ Número de chunks apropriado? (1-2 simples, 3-5 complexa)
□ Resposta é factualmente correta? (Sim/Não)
□ Linguagem é clara? (Sim/Não)
□ Tempo de resposta < 300ms? (Sim/Não)
```

---

## 📊 SEQUÊNCIA DE TESTE RECOMENDADA

### Fase 1: Validação Básica (5 min)
1. "Qual foi a inflação em julho de 2025?"
2. "O que é avaliação de políticas públicas?"
3. "Quais setores econômicos do Paraná?"
4. "Informação que não existe no PDF"
5. "Pergunta ambígua"

**Resultado esperado:** 3 respostas boas, 1 "não encontrei" correto, 1 lida com ambiguidade

### Fase 2: Testes Complexos (10 min)
1. "Como a inflação afeta desenvolvimento regional?"
2. "Quais políticas são recomendadas para o Paraná?"
3. "Descreva a situação econômica e políticas do Brasil"
4. "Qual é a relação entre indicadores econômicos e políticas?"

**Resultado esperado:** Respostas com 3-5 chunks, bem integradas

### Fase 3: Ajustes Finos (5 min)
Se tiver problemas:
- Muitos "não encontrei"? → Diminua threshold de 0.55 para 0.50
- Respostas genéricas? → Aumente threshold para 0.60
- Poucos chunks? → Aumente top_k de 10 para 12-15

---

## 🎓 EXEMPLOS DE RESPOSTAS ESPERADAS

### ✅ Resposta BOA
```
Pergunta: "Qual foi a inflação em julho de 2025?"

Resposta:
"A inflação acumulada em 12 meses atingiu 4.5% em julho de 2025, refletindo 
pressões de demanda agregada e ajustes de preços administrados. [Fonte 1, Página 5]"

Chunks: 1 (score 0.85)
⏱ Latência: 120ms
```

### ⚠️ Resposta MARGINAL
```
Pergunta: "Como a inflação afeta o desenvolvimento?"

Resposta:
"A inflação pode impactar investimentos em desenvolvimento."

Chunks: 1 (score 0.55) ← Muito poucos!
⚡ PROBLEMA: Deveria ter 3-4 chunks explicando o impacto
```

### ❌ Resposta RUIM
```
Pergunta: "Qual foi a inflação em julho de 2025?"

Resposta:
"Não encontrei essa informação nos documentos fornecidos."

Chunks: 0
⚡ PROBLEMA: A informação EXISTE no documento!
→ Aumentar threshold ou ajustar similiar
```

---

## 🚀 ROTEIRO FINAL DE TESTE

**Antes de considerar o sistema "pronto":**

1. ✅ Teste Fase 1 → Deve ter 80%+ sucesso
2. ✅ Teste Fase 2 → Deve ter 3+ chunks em respostas complexas
3. ✅ Teste Fase 3 → Afinar parâmetros se necessário
4. ✅ Teste varias perguntas do seu domínio específico

Se passar em todos, seu RAG está **production-ready** 🎉

---

## 💡 DICAS PARA TESTES

### Para obter respostas melhores:
- ✅ Pergunte em português claro (como falaria com uma pessoa)
- ✅ Use termos que aparecem nos PDFs
- ✅ Seja específico ("inflação em julho 2025" vs "inflação")
- ✅ Faça perguntas progressivamente mais complexas

### Para debugar problemas:
- 🔍 Veja os chunks recuperados no painel "Raciocínio"
- 🔍 Clique em "Mostrar prompt completo do LLM"
- 🔍 Ajuste threshold/top-k e teste novamente
- 🔍 Se chunks estão certos mas resposta ruim → problema do LLM
- 🔍 Se chunks estão ruins → problema do embedding/threshold
