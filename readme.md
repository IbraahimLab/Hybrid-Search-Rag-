User query
   ↓
BM25 retrieval ─┐
                ├─→ RRF → top-k documents
Vector retrieval┘
   ↓
(reranker)
   ↓
LLM
