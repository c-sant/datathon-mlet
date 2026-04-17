#!/usr/bin/env python3
"""
Teste RAG OFFLINE - Funciona sem conexão/cache do Hugging Face
Demonstra: Ingestão → Chunking → Embedding → Retrieval → Generation
"""

import sys
from pathlib import Path

import faiss
import numpy as np

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 🔹 DADOS MOCKADOS (Para teste sem dependências externas)
MOCK_CHUNKS = [
    "Renda Fixa Descomplicada: Entenda como aproveitar a alta dos juros para multiplicar o seu dinheiro. Com taxas acima de 10% ao ano, investimentos em renda fixa podem ser estratégicos em 2026.",
    "O mercado de ações brasileiro apresenta alternativas interessantes. Ações de empresas com dividendos altos e histórico de lucratividade continuam atrativas para investidores conservadores.",
    "ETFs de índices internacionais podem diversificar sua carteira. Com a volatilidade do mercado global, investir em fundos diversificados reduz riscos específicos.",
    "Criptomoedas: Investimento de risco? Bitcoin e Ethereum continuam sendo ativos especulativos. Recomenda-se alocar apenas uma pequena porcentagem da carteira neste segmento.",
    "Imóveis continuam sendo bons investimentos de longo prazo. Com juros altos, financiamentos estão mais caros, mas aluguel segue competitivo.",
]

MOCK_METADATA = [
    {"id": "doc_1", "title": "Guia de Renda Fixa 2026", "source": "infomoney.com"},
    {"id": "doc_2", "title": "Ações Recomendadas para 2026", "source": "estadao.com.br"},
    {"id": "doc_3", "title": "ETFs: Diversificação Global", "source": "seudinheiro.com.br"},
    {"id": "doc_4", "title": "Bitcoin e Criptomoedas em 2026", "source": "infomoney.com"},
    {"id": "doc_5", "title": "Mercado Imobiliário 2026", "source": "seudinheiro.com.br"},
]

# 🔹 EMBEDDINGS MOCKADOS (Vetores 384-dim simulados)
np.random.seed(42)
MOCK_EMBEDDINGS = np.random.randn(len(MOCK_CHUNKS), 384).astype(np.float32)

print("=" * 70)
print("🧪 TESTE RAG OFFLINE - Demonstração do Sistema RAG")
print("=" * 70)

# Step 1: Criar índice FAISS mockado
print("\n📦 Criando índice FAISS...")
index = faiss.IndexFlatL2(384)
index.add(MOCK_EMBEDDINGS)
print(f"✅ Índice criado com {index.ntotal} documentos")

# Step 2: Query embedding (mockado - seria gerado por SentenceTransformers)
print("\n🔍 Query: 'Quais ações estão recomendadas para 2026?'")
query_embedding = np.random.randn(1, 384).astype(np.float32)

# Step 3: Retrieval (busca nos primeiros 3 documentos)
print("\n--- Resultados do Retriever ---")
distances, indices = index.search(query_embedding, k=3)

retrieved_chunks = [MOCK_CHUNKS[i] for i in indices[0]]
retrieved_metadata = [MOCK_METADATA[i] for i in indices[0]]

for rank, (idx, dist) in enumerate(zip(indices[0], distances[0], strict=False), 1):
    meta = MOCK_METADATA[idx]
    print(f"Rank {rank} | Distância: {dist:.4f}")
    print(f"Doc: {meta['id']} - {meta['title']}")
    print(f"Texto: {MOCK_CHUNKS[idx][:100]}...")
    print()

# Step 4: Context concatenation
context = "\n".join([f"- {chunk}" for chunk in retrieved_chunks])

# Step 5: Generation (mockado)
print("\n--- Resposta Final (Mockada) ---")
answer = """
Para 2026, baseado nas informações financeiras mais recentes, as seguintes ações são recomendadas:

1. **Renda Fixa**: Com juros acima de 10% ao ano, títulos e fundos de renda fixa são muito atrativos. 
   CDB, tesouro direto e debêntures oferecem bom retorno com risco controlado.

2. **Ações com Dividendos**: Empresas grandes e lucrativas (blue chips) com histórico de 
   distribuição de dividendos são seguras para investidores conservadores.

3. **ETFs Internacionais**: Diversificar em fundos de índices globais reduz risco concentrado 
   no mercado brasileiro.

4. **Imóveis**: Continuar como ativo de longo prazo, pois aluguel permanece competitivo mesmo 
   com juros altos.

5. **Criptomoedas**: Apenas para investidores com boa tolerância ao risco e pequena alocação 
   nos ativos especulativos.

Recomendação geral: Manter uma carteira diversificada (60% renda fixa, 30% ações, 10% outros).
"""
print(answer)

print("\n" + "=" * 70)
print("✅ Teste RAG Offline Concluído com Sucesso!")
print("=" * 70)
print("\n📚 Para usar com dados reais:")
print("   1. Execute: python run_local.py")
print("   2. Ou defina: export HF_HUB_OFFLINE=1 (para usar cache existente)")
print("   3. Ou configure: RAG_MODEL=distilgpt2 python run_local.py")
print("=" * 70)
