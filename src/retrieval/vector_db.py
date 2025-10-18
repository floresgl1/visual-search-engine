"""Vector Database Module for RAG"""

import chromadb
import numpy as np
from typing import List, Dict, Union, Optional
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    """Vector database for image embedding storage and retrieval"""
    
    def __init__(self, persist_directory: Union[str, Path] = "./chroma_db", 
                 collection_name: str = "image_embeddings"):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        self.persist_directory.mkdir(exist_ok=True, parents=True)
        
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Visual search engine image embeddings"}
        )
        
        logger.info(f"Vector database initialized at {self.persist_directory}")
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict], 
                      ids: Optional[List[str]] = None):
        n_embeddings = len(embeddings)
        
        if ids is None:
            ids = [f"image_{i:06d}" for i in range(n_embeddings)]
        
        embeddings_list = embeddings.tolist()
        
        self.collection.add(
            embeddings=embeddings_list,
            metadatas=metadata,
            ids=ids
        )
        
        logger.info(f"Added {n_embeddings} embeddings")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
              where: Optional[Dict] = None):
        if query_embedding.ndim == 2:
            query_embedding = query_embedding.squeeze()
        
        query_list = query_embedding.tolist()
        
        results = self.collection.query(
            query_embeddings=[query_list],
            n_results=top_k,
            where=where
        )
        
        return {
            'ids': results['ids'][0],
            'distances': results['distances'][0],
            'metadatas': results['metadatas'][0]
        }
    
    def count(self):
        return self.collection.count()
    
    def get_stats(self):
        count = self.count()
        stats = {
            'collection_name': self.collection_name,
            'total_embeddings': count,
            'persist_directory': str(self.persist_directory)
        }
        
        if count > 0:
            sample = self.collection.get(limit=1, include=['embeddings'])
            if sample['embeddings']:
                stats['embedding_dimension'] = len(sample['embeddings'][0])
        
        return stats
