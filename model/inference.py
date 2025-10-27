#!/usr/bin/env python3
"""
Inference Script for Trained BGE-M3 Projection Model

Load trained model and perform embedding-based search/matching
"""

import torch
from pathlib import Path
from typing import List, Tuple, Optional
import json
from model import BGEM3WithHead


class RentalSearchEngine:
    """
    Search engine using trained BGE-M3 projection model
    
    Usage:
        # Load model
        engine = RentalSearchEngine("checkpoints/best_model.pt")
        
        # Search
        results = engine.search("phòng trọ q10 25m2 5tr5", top_k=5)
    """
    
    def __init__(
        self, 
        model_path: str,
        device: str = "auto",
        d_out: int = 128,
        freeze_encoder: bool = True,
        use_layernorm: bool = False
    ):
        """
        Initialize search engine with trained model
        
        Args:
            model_path: Path to .pt checkpoint file
            device: "cuda", "cpu", or "auto"
            d_out: Output dimension (must match training config)
            freeze_encoder: Whether encoder was frozen during training
            use_layernorm: Whether LayerNorm was used
        """
        self.device = self._setup_device(device)
        self.model_path = Path(model_path)
        
        # Load model
        print(f"🔄 Loading model from: {self.model_path}")
        self.model = self._load_model(d_out, freeze_encoder, use_layernorm)
        print(f"✅ Model loaded successfully!")
        print(f"   Device: {self.device}")
        print(f"   Output dim: {d_out}")
        
        # Pre-encoded database (optional)
        self.database_embeddings = None
        self.database_texts = None
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    def _load_model(
        self, 
        d_out: int, 
        freeze_encoder: bool, 
        use_layernorm: bool
    ) -> BGEM3WithHead:
        """Load trained model from checkpoint"""
        
        # Initialize model architecture
        model = BGEM3WithHead(
            d_out=d_out,
            freeze_encoder=freeze_encoder,
            use_layernorm=use_layernorm
        ).to(self.device)
        
        # Load checkpoint
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.model_path}\n"
                f"Available checkpoints in {self.model_path.parent}:\n"
                f"  {list(self.model_path.parent.glob('*.pt'))}"
            )
        
        # Check if it's a full checkpoint or just state_dict
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Full checkpoint (with optimizer, epoch, etc.)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"   Checkpoint loss: {checkpoint.get('loss', 'unknown'):.4f}")
        else:
            # Just state_dict (final model)
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def encode(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode texts to embeddings
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            
        Returns:
            Tensor of shape (len(texts), d_out)
        """
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                embeddings = self.model(batch_texts, device=self.device)
                all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def index_database(self, texts: List[str], batch_size: int = 32):
        """
        Pre-encode database for fast search
        
        Args:
            texts: List of property descriptions
            batch_size: Batch size for encoding
        """
        print(f"\n🔄 Indexing {len(texts)} properties...")
        self.database_texts = texts
        self.database_embeddings = self.encode(texts, batch_size)
        print(f"✅ Database indexed!")
        print(f"   Shape: {self.database_embeddings.shape}")
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        return_scores: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Search for most similar properties
        
        Args:
            query: Search query text
            top_k: Number of results to return
            return_scores: Whether to return similarity scores
            
        Returns:
            List of (text, score) tuples if return_scores=True
            List of texts otherwise
        """
        if self.database_embeddings is None:
            raise ValueError(
                "Database not indexed! Call index_database() first."
            )
        
        # Encode query
        query_emb = self.encode([query])[0]  # (d_out,)
        
        # Compute similarities
        similarities = query_emb @ self.database_embeddings.T  # (n_docs,)
        
        # Get top-k
        top_k = min(top_k, len(similarities))
        scores, indices = torch.topk(similarities, k=top_k)
        
        # Format results
        results = []
        for idx, score in zip(indices.tolist(), scores.tolist()):
            text = self.database_texts[idx]
            if return_scores:
                results.append((text, score))
            else:
                results.append(text)
        
        return results
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (cosine similarity)
        """
        embs = self.encode([text1, text2])
        similarity = (embs[0] @ embs[1]).item()
        return similarity


def demo_usage():
    """Demo: How to use the inference script"""
    
    print("=" * 80)
    print("🚀 BGE-M3 Inference Demo")
    print("=" * 80)
    
    # 1. Load trained model
    model_path = "checkpoints/best_model.pt"
    
    try:
        engine = RentalSearchEngine(
            model_path=model_path,
            device="auto",
            d_out=128,  # Must match training config!
            freeze_encoder=True,
            use_layernorm=False
        )
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\n💡 Train a model first:")
        print("   python train_script.py")
        return
    
    # 2. Create a sample database
    print("\n" + "=" * 80)
    print("📚 Creating Sample Database")
    print("=" * 80)
    
    database = [
        "Phòng trọ 25m2 Quận 10, WC riêng, máy lạnh, giá 5.5 triệu/tháng",
        "Cho thuê phòng 30m2 Quận 1, full nội thất, giá 8 triệu/tháng",
        "Phòng 20m2 Thủ Đức, WC chung, giá 3.5 triệu/tháng",
        "Studio 35m2 Quận 3, ban công, bếp riêng, giá 9 triệu/tháng",
        "Phòng 15m2 Bình Thạnh, giá rẻ 2.5 triệu/tháng",
    ]
    
    engine.index_database(database)
    
    # 3. Search examples
    print("\n" + "=" * 80)
    print("🔍 Search Examples")
    print("=" * 80)
    
    queries = [
        "phòng trọ q10 25m2 wc riêng 5tr5",
        "thuê phòng q1 đầy đủ nội thất",
        "phòng giá rẻ dưới 3 triệu",
    ]
    
    for query in queries:
        print(f"\n🔎 Query: \"{query}\"")
        print("-" * 80)
        
        results = engine.search(query, top_k=3)
        
        for rank, (text, score) in enumerate(results, 1):
            print(f"   {rank}. [{score:.4f}] {text}")
    
    # 4. Similarity computation
    print("\n" + "=" * 80)
    print("🔗 Pairwise Similarity")
    print("=" * 80)
    
    text1 = "phòng trọ quận 10 wc riêng"
    text2 = "Phòng trọ 25m2 Quận 10, WC riêng, máy lạnh"
    
    similarity = engine.compute_similarity(text1, text2)
    print(f"\nText 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Similarity: {similarity:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    demo_usage()

