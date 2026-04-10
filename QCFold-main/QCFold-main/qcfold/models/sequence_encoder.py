"""
Sequence encoder module.

Supports:
  1. ESM-2 pre-trained protein language model (primary)
  2. One-hot encoding fallback (for CPU-only environments)
  3. BLOSUM62 encoding
  4. MSA feature extraction (when MSA is available)

The encoder produces per-residue feature vectors that capture
evolutionary and structural information from the protein sequence.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


# Standard amino acid alphabet
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}

# BLOSUM62 scoring matrix (simplified, 20x20)
BLOSUM62 = np.array([
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
], dtype=np.float32)


class OneHotEncoder(nn.Module):
    """Simple one-hot + BLOSUM62 encoding for protein sequences."""
    
    def __init__(self, embed_dim: int = 320):
        super().__init__()
        self.embed_dim = embed_dim
        # One-hot (20) + BLOSUM62 row (20) = 40 features
        self.projection = nn.Linear(40, embed_dim)
    
    def forward(self, sequence: str) -> torch.Tensor:
        """Encode a protein sequence.
        
        Args:
            sequence: amino acid string
        Returns:
            (L, embed_dim) tensor
        """
        L = len(sequence)
        features = np.zeros((L, 40), dtype=np.float32)
        
        for i, aa in enumerate(sequence):
            idx = AA_TO_IDX.get(aa, -1)
            if idx >= 0:
                features[i, idx] = 1.0  # One-hot
                features[i, 20:] = BLOSUM62[idx]  # BLOSUM62 row
        
        x = torch.tensor(features)
        return self.projection(x)


class ESM2Encoder(nn.Module):
    """ESM-2 protein language model encoder.
    
    Uses the smallest ESM-2 model (8M parameters) for CPU inference.
    Produces rich per-residue embeddings trained on millions of sequences.
    """
    
    def __init__(
        self,
        model_name: str = "esm2_t6_8M_UR50D",
        embed_dim: int = 320,
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.freeze = freeze
        self._model = None
        self._alphabet = None
        self._batch_converter = None
    
    def _load_model(self):
        """Lazy-load ESM-2 model."""
        if self._model is not None:
            return
        
        try:
            import esm
            self._model, self._alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self._batch_converter = self._alphabet.get_batch_converter()
            if self.freeze:
                for param in self._model.parameters():
                    param.requires_grad = False
                self._model.eval()
        except (ImportError, Exception):
            # Fallback: ESM not available, use one-hot
            self._model = None
            self._fallback = OneHotEncoder(self.embed_dim)
    
    def forward(self, sequence: str) -> torch.Tensor:
        """Encode a protein sequence with ESM-2.
        
        Args:
            sequence: amino acid string
        Returns:
            (L, embed_dim) tensor
        """
        self._load_model()
        
        if self._model is None:
            return self._fallback(sequence)
        
        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = self._batch_converter(data)
        
        with torch.no_grad():
            results = self._model(batch_tokens, repr_layers=[6])
        
        # Extract per-residue representations (skip BOS/EOS tokens)
        representations = results["representations"][6]
        return representations[0, 1:len(sequence)+1, :]


def get_encoder(encoder_type: str = "onehot", **kwargs) -> nn.Module:
    """Factory function for sequence encoders."""
    if encoder_type == "esm2":
        return ESM2Encoder(**kwargs)
    else:
        return OneHotEncoder(**kwargs)
