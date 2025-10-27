# pip install transformers accelerate peft
from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "BAAI/bge-m3" # -> 1024 dim
D_IN = 1024
D_OUT = 128  # -> 128
TAU = 0.07  # temperature cho InfoNCE


def mean_pool(last_hidden_state, attention_mask):
    # last_hidden_state: [B, T, H], attention_mask: [B, T]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B,T,1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B,H]
    counts = mask.sum(dim=1).clamp(min=1e-6)                        # [B,1]
    return summed / counts


class ProjectionHead(nn.Module):
    def __init__(self, d_in: int, d_out: int, use_layernorm: bool = False):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=False)
        self.ln = nn.LayerNorm(d_out) if use_layernorm else None

        # (Tuỳ chọn) PCA init có thể làm riêng rồi copy weight vào self.linear.weight.data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.ln is not None:
            x = self.ln(x)
        # L2Norm để dùng cosine/dot tương đương
        x = F.normalize(x, p=2, dim=-1)
        return x


class BGEM3WithHead(nn.Module):
    def __init__(self, d_out: int = D_OUT, freeze_encoder: bool = True, use_layernorm=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, use_fast=True)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = ProjectionHead(D_IN, d_out, use_layernorm=use_layernorm)

    @torch.no_grad()
    def encode_text_1024(self, texts: List[str], max_length: int = 8192, device=None) -> torch.Tensor:
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        out = self.encoder(**enc)
        pooled = mean_pool(out.last_hidden_state,
                           enc["attention_mask"])  # [B, 1024]
        return pooled

    def forward(self, texts: List[str], max_length: int = 512, device=None) -> torch.Tensor:
        """
        Forward pass: encode texts to embeddings
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length (default: 512)
            device: Target device (cuda/cpu)
            
        Returns:
            L2-normalized embeddings [B, d_out] (default d_out=128)
        """
        with torch.set_grad_enabled(any(p.requires_grad for p in self.encoder.parameters())):
            enc = self.tokenizer(
                texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            )
            if device is not None:
                enc = {k: v.to(device) for k, v in enc.items()}
            out = self.encoder(**enc)
            pooled = mean_pool(out.last_hidden_state,
                               enc["attention_mask"])  # [B,1024]
        return self.head(pooled)  # [B, d_out] (L2-normalized)
