from fastapi import FastAPI
from rag.ingest import embedder, index, docs
from rag.retriever import retrieve
from rag.generator import generate_answer

app = FastAPI()

@app.get("/query")
def query_rag(q: str):
    results = retrieve(q, embedder, index, docs)
    context = " ".join([r["text"] for r in results])
    answer = generate_answer(q, context)
    return {"query": q, "context": context, "answer": answer}