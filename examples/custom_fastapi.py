import os
import shutil
import tempfile
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from apex_rag import ApexIndex

# Global reference to our index
index: ApexIndex = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global index
    print("🚀 Initializing ApexRAG Engine...")
    # Initialize the index connected to a local SQLite DB
    # We use llama3.1:8b to match your Ollama setup!
    index = await ApexIndex.create(
        db_url="sqlite+aiosqlite:///my_api_database.db",
        model="llama3.1:8b",
        verify_leaves=True
    )
    yield
    # Cleanup on shutdown
    await index.close()

# Initialize FastAPI with the lifespan
app = FastAPI(title="My Custom ApexRAG API", lifespan=lifespan)

# --- Pydantic Models for our API ---
class QueryRequest(BaseModel):
    doc_id: str
    question: str

# --- 1. Upload & Ingest Endpoint ---
@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...), 
    doc_id: str = Form(...)  # Let the user pick a custom ID like "my_book"
):
    """Uploads a PDF, TXT, or DOCX and processes it into the Agentic Tree."""
    if not index:
        raise HTTPException(status_code=503, detail="Index not ready")

    # Save the uploaded file to a temporary location
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        print(f"📥 Ingesting {file.filename} into ApexRAG...")
        # Ingest the file using ApexRAG
        ingested_id = await index.ingest(
            file_path=tmp_path,
            doc_id=doc_id,
            synthesize_summaries=True
        )
        return {"status": "success", "message": "Document indexed!", "doc_id": ingested_id}
    finally:
        # Clean up the temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/export/{doc_id}")
async def export_document_tree(doc_id: str):
    """Export the exact document hierarchy as a nested JSON tree (PageIndex style)."""
    if not index:
        raise HTTPException(status_code=503, detail="Index not ready")
        
    tree = await index.export_tree(doc_id)
    if not tree:
        return {"error": "Document not found or empty."}
    return tree

# --- 3. Query Endpoint ---
@app.post("/ask")
async def ask_question(request: QueryRequest):
    """Ask a question about a specific document."""
    if not index:
        raise HTTPException(status_code=503, detail="Index not ready")

    print(f"🔍 Searching {request.doc_id} for: '{request.question}'")
    
    # Run the Agentic Query
    result = await index.query(request.question, request.doc_id)

    if not result:
        return {"found": False, "answer": "Could not find an answer in the document."}

    return {
        "found": True,
        "verified": result.verified,
        "confidence": result.confidence,
        "path": result.path,
        "answer": result.content,
        "trace": result.trace
    }

if __name__ == "__main__":
    # Run the server on port 8080
    uvicorn.run("custom_fastapi:app", host="0.0.0.0", port=8080, reload=True)
