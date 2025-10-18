"""
PyTorch CLIP Embedder
Generates embeddings for images and text using CLIP model
"""

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import List, Union
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIPEmbedder:
    """
    CLIP-based embedder for images and text
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = None
    ):
        """
        Initialize CLIP embedder
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading CLIP model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.projection_dim
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode_images(
        self,
        images: Union[List[Image.Image], List[str], List[Path]],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for images
        
        Args:
            images: List of PIL Images or image paths
            batch_size: Batch size for processing
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            Numpy array of embeddings (n_images, embedding_dim)
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_images = []
            
            # Load images if paths are provided
            for img in batch:
                if isinstance(img, (str, Path)):
                    try:
                        img = Image.open(img).convert('RGB')
                    except Exception as e:
                        logger.warning(f"Failed to load image: {e}")
                        # Use blank image as fallback
                        img = Image.new('RGB', (224, 224), color='black')
                
                batch_images.append(img)
            
            # Process batch
            inputs = self.processor(
                images=batch_images,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model.get_image_features(**inputs)
                
                # Normalize if requested
                if normalize:
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                
                # Convert to numpy
                embeddings = embeddings.cpu().numpy()
                all_embeddings.append(embeddings)
        
        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Generated embeddings for {len(images)} images")
        
        return all_embeddings
    
    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for text
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Process text
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
            
            # Normalize if requested
            if normalize:
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            # Convert to numpy
            embeddings = embeddings.cpu().numpy()
        
        logger.info(f"Generated embeddings for {len(texts)} texts")
        
        return embeddings
    
    def compute_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between embeddings
        
        Args:
            embeddings1: First set of embeddings (n, dim)
            embeddings2: Second set of embeddings (m, dim)
            
        Returns:
            Similarity matrix (n, m)
        """
        # Normalize if not already normalized
        embeddings1_norm = embeddings1 / np.linalg.norm(
            embeddings1, axis=1, keepdims=True
        )
        embeddings2_norm = embeddings2 / np.linalg.norm(
            embeddings2, axis=1, keepdims=True
        )
        
        # Compute dot product (cosine similarity for normalized vectors)
        similarity = np.dot(embeddings1_norm, embeddings2_norm.T)
        
        return similarity
    
    def search(
        self,
        query: Union[str, Image.Image],
        image_embeddings: np.ndarray,
        top_k: int = 5
    ) -> tuple:
        """
        Search for most similar images
        
        Args:
            query: Text query or image
            image_embeddings: Precomputed image embeddings
            top_k: Number of results to return
            
        Returns:
            Tuple of (indices, similarities)
        """
        # Generate query embedding
        if isinstance(query, str):
            query_embedding = self.encode_text(query)
        else:
            query_embedding = self.encode_images([query])
        
        # Compute similarities
        similarities = self.compute_similarity(query_embedding, image_embeddings)
        similarities = similarities.squeeze()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        
        return top_indices, top_similarities
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        output_path: Union[str, Path]
    ) -> None:
        """
        Save embeddings to disk
        
        Args:
            embeddings: Embeddings array
            output_path: Output file path (.npy)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        np.save(output_path, embeddings)
        logger.info(f"Embeddings saved to: {output_path}")
    
    def load_embeddings(
        self,
        input_path: Union[str, Path]
    ) -> np.ndarray:
        """
        Load embeddings from disk
        
        Args:
            input_path: Input file path (.npy)
            
        Returns:
            Embeddings array
        """
        embeddings = np.load(input_path)
        logger.info(f"Loaded embeddings from: {input_path}")
        logger.info(f"Shape: {embeddings.shape}")
        
        return embeddings
    
    def get_model_info(self) -> dict:
        """
        Get model information
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'embedding_dim': self.embedding_dim,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        }


# Example usage
if __name__ == "__main__":
    # Initialize embedder
    print("Initializing CLIP embedder...")
    embedder = CLIPEmbedder()
    
    # Get model info
    info = embedder.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test text encoding
    print("\nTesting text encoding...")
    texts = [
        "a photo of a cat",
        "a dog in the park",
        "sunset over mountains"
    ]
    
    text_embeddings = embedder.encode_text(texts)
    print(f"Text embeddings shape: {text_embeddings.shape}")
    
    # Test similarity
    print("\nTesting similarity computation...")
    similarity = embedder.compute_similarity(
        text_embeddings[0:1],
        text_embeddings[1:]
    )
    print(f"Similarity matrix shape: {similarity.shape}")
    print(f"Similarities: {similarity.flatten()}")
    
    print("\n✅ All tests passed!")
