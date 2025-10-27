import torch
from typing import List
from loguru import logger
from transformers import AutoModel
from config.settings import settings

class EmbeddingService:
    def __init__(self):
        self.device = settings.DEVICE if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading custom BGE-M3 model from Hugging Face on {self.device}...")
        
        # Load model from Hugging Face
        # Model: https://huggingface.co/lamdx4/bge-m3-vietnamese-rental-projection
        self.model = AutoModel.from_pretrained(
            settings.EMBEDDING_MODEL,
            trust_remote_code=True
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("✅ Custom BGE-M3 projection model loaded from Hugging Face")
        logger.info(f"   Model: {settings.EMBEDDING_MODEL}")
        logger.info(f"   Output dimension: {settings.EMBEDDING_DIM}")
        logger.info(f"   Device: {self.device}")
    
    def generate_dense_embedding(self, text: str, dim: int = None) -> List[float]:
        """
        Generate dense embedding using trained projection model
        
        Args:
            text: Input text
            dim: Output dimension (fixed at 128 for this model)
        
        Returns:
            List of floats (L2-normalized embedding vector, 128-dim)
        """
        # Use model's encode method (from HF model card)
        with torch.no_grad():
            embedding = self.model.encode([text], device=self.device)  # [1, 128]
            embedding = embedding[0].cpu().tolist()  # Convert to list
        
        return embedding

# Singleton instance
_embedding_service = None

def get_embedding_service():
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

