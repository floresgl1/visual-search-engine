"""
Vector Database Module for RAG
Implements vector storage and retrieval using ChromaDB
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Union, Optional
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    Vector database for image embedding storage and retrieval
    """
    
    def __init__(
        self,
        persist_directory: Union[str, Path] = "./chroma_db",
        collection_name: str = "image_embeddings"
    ):
        """
        Initialize vector database
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        self.persist_directory.mkdir(exist_ok=True, parents=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Visual search engine image embeddings"}
        )
        
        logger.info(f"Vector database initialized")
        logger.info(f"Persist directory: {self.persist_directory}")
        logger.info(f"Collection: {self.collection_name}")
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add embeddings to the database
        
        Args:
            embeddings: Embedding vectors (n, dim)
            metadata: List of metadata dictionaries for each embedding
            ids: Optional list of IDs (will generate if not provided)
        """
        n_embeddings = len(embeddings)
        
        if len(metadata) != n_embeddings:
            raise ValueError(
                f"Metadata length ({len(metadata)}) must match "
                f"embeddings length ({n_embeddings})"
            )
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"image_{i:06d}" for i in range(n_embeddings)]
        
        # Convert embeddings to list format
        embeddings_list = embeddings.tolist()
        
        # Add to collection in batches (ChromaDB recommends < 41666 per batch)
        batch_size = 40000
        for i in range(0, n_embeddings, batch_size):
            end_idx = min(i + batch_size, n_embeddings)
            
            self.collection.add(
                embeddings=embeddings_list[i:end_idx],
                metadatas=metadata[i:end_idx],
                ids=ids[i:end_idx]
            )
            
            logger.info(
                f"Added embeddings {i} to {end_idx} "
                f"({end_idx}/{n_embeddings})"
            )
        
        logger.info(f"Successfully added {n_embeddings} embeddings")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding vector (1, dim) or (dim,)
            top_k: Number of results to return
            where: Optional metadata filter
            
        Returns:
            Dictionary with ids, distances, and metadatas
        """
        # Ensure query embedding is 1D
        if query_embedding.ndim == 2:
            query_embedding = query_embedding.squeeze()
        
        # Convert to list
        query_list = query_embedding.tolist()
        
        # Query the collection
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
    
    def search_by_metadata(
        self,
        where: Dict,
        top_k: int = 10
    ) -> Dict:
        """
        Search by metadata filters
        
        Args:
            where: Metadata filter (e.g., {"category": "nature"})
            top_k: Maximum number of results
            
        Returns:
            Dictionary with ids and metadatas
        """
        results = self.collection.get(
            where=where,
            limit=top_k
        )
        
        return {
            'ids': results['ids'],
            'metadatas': results['metadatas']
        }
    
    def get_by_ids(
        self,
        ids: List[str]
    ) -> Dict:
        """
        Get embeddings by IDs
        
        Args:
            ids: List of embedding IDs
            
        Returns:
            Dictionary with embeddings and metadatas
        """
        results = self.collection.get(
            ids=ids,
            include=['embeddings', 'metadatas']
        )
        
        return {
            'embeddings': np.array(results['embeddings']),
            'metadatas': results['metadatas']
        }
    
    def update_metadata(
        self,
        ids: List[str],
        metadatas: List[Dict]
    ) -> None:
        """
        Update metadata for existing embeddings
        
        Args:
            ids: List of embedding IDs
            metadatas: List of new metadata dictionaries
        """
        self.collection.update(
            ids=ids,
            metadatas=metadatas
        )
        
        logger.info(f"Updated metadata for {len(ids)} embeddings")
    
    def delete(
        self,
        ids: List[str]
    ) -> None:
        """
        Delete embeddings by IDs
        
        Args:
            ids: List of embedding IDs to delete
        """
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} embeddings")
    
    def count(self) -> int:
        """
        Get total number of embeddings
        
        Returns:
            Count of embeddings in the database
        """
        return self.collection.count()
    
    def peek(self, limit: int = 5) -> Dict:
        """
        Peek at first few embeddings
        
        Args:
            limit: Number of embeddings to peek
            
        Returns:
            Dictionary with sample data
        """
        results = self.collection.peek(limit=limit)
        return {
            'ids': results['ids'],
            'metadatas': results['metadatas']
        }
    
    def reset(self) -> None:
        """
        Reset the collection (delete all data)
        """
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Visual search engine image embeddings"}
        )
        logger.info("Collection reset")
    
    def export_to_file(
        self,
        output_path: Union[str, Path]
    ) -> None:
        """
        Export embeddings and metadata to file
        
        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Get all data
        all_data = self.collection.get(
            include=['embeddings', 'metadatas']
        )
        
        # Convert to serializable format
        export_data = {
            'ids': all_data['ids'],
            'embeddings': np.array(all_data['embeddings']).tolist(),
            'metadatas': all_data['metadatas'],
            'count': len(all_data['ids'])
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {export_data['count']} embeddings to {output_path}")
    
    def import_from_file(
        self,
        input_path: Union[str, Path]
    ) -> None:
        """
        Import embeddings and metadata from file
        
        Args:
            input_path: Input file path
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        embeddings = np.array(data['embeddings'])
        
        self.add_embeddings(
            embeddings=embeddings,
            metadata=data['metadatas'],
            ids=data['ids']
        )
        
        logger.info(f"Imported {len(data['ids'])} embeddings from {input_path}")
    
    def get_stats(self) -> Dict:
        """
        Get database statistics
        
        Returns:
            Dictionary with statistics
        """
        count = self.count()
        
        stats = {
            'collection_name': self.collection_name,
            'total_embeddings': count,
            'persist_directory': str(self.persist_directory)
        }
        
        # Get sample to check embedding dimension
        if count > 0:
            sample = self.collection.get(limit=1, include=['embeddings'])
            if sample['embeddings']:
                stats['embedding_dimension'] = len(sample['embeddings'][0])
        
        return stats


# Example usage
if __name__ == "__main__":
    # Initialize database
    db = VectorDatabase(
        persist_directory="./test_chroma_db",
        collection_name="test_collection"
    )
    
    # Create sample embeddings
    n_samples = 100
    embedding_dim = 512
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create sample metadata
    categories = ['nature', 'city', 'people', 'food']
    metadata = [
        {
            'image_id': f'img_{i:03d}',
            'category': categories[i % len(categories)],
            'description': f'Sample image {i}'
        }
        for i in range(n_samples)
    ]
    
    # Add to database
    print("Adding embeddings to database...")
    db.add_embeddings(embeddings, metadata)
    
    # Get stats
    stats = db.get_stats()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test search
    query_embedding = np.random.randn(1, embedding_dim).astype(np.float32)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    print("\nSearching...")
    results = db.search(query_embedding, top_k=5)
    
    print(f"\nTop 5 results:")
    for i, (id_, dist, meta) in enumerate(
        zip(results['ids'], results['distances'], results['metadatas'])
    ):
        print(f"{i+1}. ID: {id_}, Distance: {dist:.4f}, Category: {meta['category']}")
    
    # Test metadata filter
    print("\nFiltering by category='nature':")
    filtered = db.search_by_metadata(where={"category": "nature"}, top_k=5)
    print(f"Found {len(filtered['ids'])} results")
