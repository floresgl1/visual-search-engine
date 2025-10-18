"""
Visual Search Engine API
FastAPI application for visual search
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from PIL import Image
import io
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.embeddings.pytorch_embedder import CLIPEmbedder
from src.retrieval.vector_db import VectorDatabase

# Initialize FastAPI app
app = FastAPI(
    title="Visual Search Engine API",
    description="Search through images using natural language queries",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    id: str
    similarity: float
    metadata: dict

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int

# Global variables
embedder: Optional[CLIPEmbedder] = None
vector_db: Optional[VectorDatabase] = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and database on startup"""
    global embedder, vector_db
    
    print("🚀 Starting Visual Search Engine API...")
    
    # Initialize CLIP embedder
    print("Loading CLIP model...")
    embedder = CLIPEmbedder(model_name="openai/clip-vit-base-patch32")
    print("✅ CLIP model loaded")
    
    # Initialize vector database
    print("Connecting to vector database...")
    # get abosulte path to chroma_db
    from pathlib import Path 
    project_root = Path(__file__).parent.parent.parent
    chroma_path = project_root / "chroma_db"

    print(f"ChromaDB path: {chroma_path.resolve()}")

    vector_db = VectorDatabase(
      persist_directory=str(chroma_path),
      collection_name="image_embeddings"

    )
    print(f"✅ Vector database connected ({vector_db.count()} embeddings)")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Visual Search Engine API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if embedder is None or vector_db is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "status": "healthy",
        "database_count": vector_db.count(),
        "model_loaded": True
    }

@app.post("/search/text", response_model=SearchResponse)
async def search_by_text(request: TextSearchRequest):
    """Search for images using a text query"""
    import time
    
    if embedder is None or vector_db is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    start_time = time.time()
    
    try:
        # Generate query embedding
        query_embedding = embedder.encode_text(request.query)
        
        # Search in vector database
        results = vector_db.search(
            query_embedding=query_embedding,
            top_k=request.top_k
        )
        
        # Format results
        search_results = []
        for id_, dist, meta in zip(
            results['ids'],
            results['distances'],
            results['metadatas']
        ):
            search_results.append(SearchResult(
                id=id_,
                similarity=float(1 - dist),
                metadata=meta
            ))
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=50)
):
    """Search for similar images using an uploaded image"""
    
    if embedder is None or vector_db is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Read and process uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Generate image embedding
        image_embedding = embedder.encode_images([image])
        
        # Search in vector database
        results = vector_db.search(
            query_embedding=image_embedding,
            top_k=top_k
        )
        
        # Format results
        search_results = []
        for id_, dist, meta in zip(
            results['ids'],
            results['distances'],
            results['metadatas']
        ):
            search_results.append(SearchResult(
                id=id_,
                similarity=float(1 - dist),
                metadata=meta
            ))
        
        return SearchResponse(
            query=f"image:{file.filename}",
            results=search_results,
            total_results=len(search_results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
