from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rag.ingest import ingest_documents, embedder, index, all_chunks, metadata
from rag.retriever import retrieve
from rag.generator import generate_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


class Document(BaseModel):
    id: str | None = Field(None, description="Identificador único do documento")
    title: str | None = Field(None, description="Título do documento")
    text: str = Field(..., description="Conteúdo em texto do documento")


class IngestRequest(BaseModel):
    docs: list[Document] = Field(..., description="Lista de documentos a serem ingeridos")
    overwrite: bool = Field(True, description="Substituir a base existente se True")


@app.post("/ingest")
def ingest_rag(payload: IngestRequest):
    try:
        stats = ingest_documents(
            [doc.model_dump() for doc in payload.docs],
            overwrite=payload.overwrite,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "status": "ok",
        "doc_count": stats["doc_count"],
        "chunk_count": stats["chunk_count"],
        "overwrite": payload.overwrite,
    }


@app.get("/query")
def query_rag(q: str, top_k: int = 3):
    if index is None or len(all_chunks) == 0:
        raise HTTPException(status_code=503, detail="Índice de busca não está disponível.")

    results = retrieve(q, embedder, index, all_chunks, metadata, top_k=top_k)
    context = " ".join([r["text"] for r in results])
    answer = generate_answer(q, context)
    return {"query": q, "top_k": top_k, "context": context, "answer": answer}