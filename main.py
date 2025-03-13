#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gemini Embedding Model Implementation
=====================================

This module implements the Gemini Embedding model as described in the research paper
"Gemini Embedding: Generalizable Embeddings from Gemini" (2025).

The Gemini Embedding model creates holistic representations of inputs for diverse 
downstream tasks, including retrieval, clustering, classification, and ranking by
leveraging the power of Gemini's transformer architecture.

Key components:
- Transformer backbone initialized from Gemini
- Mean pooling strategy for token embeddings
- Linear projection to target dimension
- Noise-contrastive estimation loss with in-batch negatives
- Multi-resolution loss (MRL) for supporting different embedding dimensions

Author: Claude
License: MIT
"""

import os
import math
import json
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Any,
    TypeVar,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from loguru import logger
from dataclasses import dataclass, field

# Type definitions
Tensor = torch.Tensor
T = TypeVar("T")

# Set up logger
logger.remove()
logger.add(
    sink=os.environ.get("LOGURU_SINK", "gemini_embedding_{time}.log"),
    level=os.environ.get("LOGURU_LEVEL", "INFO"),
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    rotation="100 MB",
    retention="1 week",
)


@dataclass
class ModelConfig:
    """Configuration for Gemini Embedding model."""

    # Model architecture parameters
    model_dim: int = 4096  # Dimension of the Gemini model (dM)
    output_dim: int = 3072  # Final embedding dimension (d)
    mrl_dims: List[int] = field(
        default_factory=lambda: [768, 1536, 3072]
    )  # Dimensions for multi-resolution loss

    # Training parameters
    batch_size: int = 128
    learning_rate: float = 1e-5
    temperature: float = (
        0.07  # Temperature parameter (τ) for similarity scaling
    )
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_train_epochs: int = 3

    # Model components
    transformer_name: str = (
        "gemini"  # Name of the transformer model to initialize from
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_file: Union[str, Path]) -> "ModelConfig":
        """Load configuration from JSON file."""
        with open(json_file, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def save_to_json(self, json_file: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(json_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with bidirectional context as used in the Gemini model.

    This implementation allows each token to attend to all other tokens in the sequence,
    implementing the bidirectional attention described in the paper.
    """

    def __init__(
        self, d_model: int, num_heads: int, dropout: float = 0.1
    ):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        assert (
            d_model % num_heads == 0
        ), "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: Tensor) -> Tensor:
        """
        Split the last dimension into (num_heads, d_k).

        Args:
            x: Input tensor [batch_size, seq_length, d_model]

        Returns:
            Tensor with shape [batch_size, num_heads, seq_length, d_k]
        """
        batch_size, seq_length, _ = x.size()
        return x.view(
            batch_size, seq_length, self.num_heads, self.d_k
        ).transpose(1, 2)

    def merge_heads(self, x: Tensor) -> Tensor:
        """
        Merge the (num_heads, d_k) back into d_model.

        Args:
            x: Input tensor [batch_size, num_heads, seq_length, d_k]

        Returns:
            Tensor with shape [batch_size, seq_length, d_model]
        """
        batch_size, _, seq_length, _ = x.size()
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, self.d_model)
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute multi-head attention.

        Args:
            query: Query tensor [batch_size, query_len, d_model]
            key: Key tensor [batch_size, key_len, d_model]
            value: Value tensor [batch_size, value_len, d_model]
            mask: Optional attention mask [batch_size, 1, 1, key_len]

        Returns:
            Output tensor [batch_size, query_len, d_model]
        """
        query.size(0)

        # Linear projections
        q = self.q_proj(query)  # [batch_size, query_len, d_model]
        k = self.k_proj(key)  # [batch_size, key_len, d_model]
        v = self.v_proj(value)  # [batch_size, value_len, d_model]

        # Split heads
        q = self.split_heads(
            q
        )  # [batch_size, num_heads, query_len, d_k]
        k = self.split_heads(
            k
        )  # [batch_size, num_heads, key_len, d_k]
        v = self.split_heads(
            v
        )  # [batch_size, num_heads, value_len, d_k]

        # Scaled dot-product attention
        # matmul: [batch_size, num_heads, query_len, key_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.d_k
        )

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # [batch_size, num_heads, query_len, d_k]
        context = torch.matmul(attn_weights, v)

        # Merge heads
        context = self.merge_heads(
            context
        )  # [batch_size, query_len, d_model]

        # Final linear projection
        output = self.output_proj(
            context
        )  # [batch_size, query_len, d_model]

        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize feed-forward network.

        Args:
            d_model: Model dimension
            d_ff: Hidden dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = (
            nn.GELU()
        )  # Using GELU activation as in modern transformers

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply feed-forward network.

        Args:
            x: Input tensor [batch_size, seq_length, d_model]

        Returns:
            Output tensor [batch_size, seq_length, d_model]
        """
        return self.linear2(
            self.dropout(self.activation(self.linear1(x)))
        )


class LayerNorm(nn.Module):
    """
    Layer normalization.
    """

    def __init__(self, d_model: int, eps: float = 1e-12):
        """
        Initialize layer normalization.

        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
        """
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply layer normalization.

        Args:
            x: Input tensor [batch_size, seq_length, d_model]

        Returns:
            Normalized tensor [batch_size, seq_length, d_model]
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + self.eps
        return self.gamma * (x - mean) / std + self.beta


class TransformerLayer(nn.Module):
    """
    Transformer layer with bidirectional attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        """
        Initialize transformer layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()

        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(
            d_model, num_heads, dropout
        )

        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Apply transformer layer.

        Args:
            x: Input tensor [batch_size, seq_length, d_model]
            mask: Optional attention mask [batch_size, 1, seq_length, seq_length]

        Returns:
            Output tensor [batch_size, seq_length, d_model]
        """
        # Self-attention with residual connection and layer normalization
        attn_output = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x), mask
        )
        x = x + self.dropout(attn_output)

        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x


class BidirectionalTransformer(nn.Module):
    """
    Transformer model with bidirectional attention as used in Gemini.

    This implementation allows each token to attend to all other tokens in the sequence,
    implementing the bidirectional attention described in the paper.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.1,
        max_seq_length: int = 2048,
    ):
        """
        Initialize transformer.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            num_layers: Number of transformer layers
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
        """
        super().__init__()

        self.d_model = d_model

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embedding
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, max_seq_length, d_model)
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        # Layer normalization
        self.norm = LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize parameters
        self._init_parameters()

        logger.info(
            f"Initialized BidirectionalTransformer with {num_layers} layers"
        )

    def _init_parameters(self):
        """
        Initialize model parameters.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Process input tokens through transformer.

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Optional attention mask [batch_size, seq_length]
                1 for tokens to attend to, 0 for tokens to ignore

        Returns:
            Token embeddings [batch_size, seq_length, d_model]
        """
        seq_length = input_ids.size(1)

        # Token embedding
        x = self.token_embedding(
            input_ids
        )  # [batch_size, seq_length, d_model]

        # Add positional embedding
        x = x + self.positional_embedding[:, :seq_length, :]

        # Apply dropout
        x = self.dropout(x)

        # Prepare attention mask if provided
        if attention_mask is not None:
            # Convert mask of shape [batch_size, seq_length] to shape [batch_size, 1, 1, seq_length]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # In the mask, 1 means "attend to" and 0 means "ignore"
            # For the attention computation, we need 1 for positions to attend to and 0 (or a large negative number after softmax) for positions to ignore
            # This creates a bidirectional mask where each token can attend to all other tokens
            attn_mask = attention_mask
        else:
            attn_mask = None

        # Process through transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask)

        # Apply final layer normalization
        x = self.norm(x)

        return x


def create_bidirectional_transformer(
    config: ModelConfig,
) -> nn.Module:
    """
    Create bidirectional transformer from scratch.

    Args:
        config: Model configuration

    Returns:
        Initialized transformer model
    """
    logger.info("Creating bidirectional transformer from scratch")

    # Define model hyperparameters
    vocab_size = (
        50304  # Example vocab size, should be configured as needed
    )
    num_heads = 16  # Number of attention heads
    d_ff = (
        4 * config.model_dim
    )  # Common practice to set d_ff to 4x model dimension
    num_layers = 12  # Number of transformer layers

    return BidirectionalTransformer(
        vocab_size=vocab_size,
        d_model=config.model_dim,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=0.1,
        max_seq_length=2048,
    )


class MeanPooler(nn.Module):
    """
    Applies mean pooling to token embeddings to create a single embedding.

    As described in the paper, simple pooling strategies can be effective in model adaptation,
    and mean pooling is chosen for the Gemini Embedding model.
    """

    def forward(
        self,
        token_embeddings: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply mean pooling to token embeddings.

        Args:
            token_embeddings: Token embeddings [batch_size, sequence_length, model_dim]
            attention_mask: Optional attention mask [batch_size, sequence_length]

        Returns:
            Pooled embeddings [batch_size, model_dim]
        """
        if attention_mask is not None:
            # Apply mask before pooling
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(
                token_embeddings * mask_expanded, dim=1
            )
            sum_mask = torch.sum(mask_expanded, dim=1)
            sum_mask = torch.clamp(
                sum_mask, min=1e-9
            )  # Prevent division by zero
            return sum_embeddings / sum_mask
        else:
            # Simple mean pooling
            return torch.mean(token_embeddings, dim=1)


class GeminiEmbedding(nn.Module):
    """
    Gemini Embedding model as described in the paper.

    The model consists of:
    1. A transformer backbone initialized from Gemini (M)
    2. Mean pooling of token embeddings (P)
    3. Linear projection to target dimension (f)
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize Gemini Embedding model.

        Args:
            config: Model configuration
        """
        super().__init__()

        self.config = config

        # Create transformer backbone from scratch with bidirectional attention
        self.transformer = create_bidirectional_transformer(config)

        # Mean pooler for token embeddings
        self.pooler = MeanPooler()

        # Linear projection to target dimension
        self.projection = nn.Linear(
            config.model_dim, config.output_dim, bias=False
        )

        # Initialize projection with small weights
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)

        logger.info(
            f"Initialized GeminiEmbedding with output dimension {config.output_dim}"
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        normalize: bool = True,
        output_dim: Optional[int] = None,
    ) -> Tensor:
        """
        Generate embeddings for input tokens.

        Args:
            input_ids: Token IDs [batch_size, sequence_length]
            attention_mask: Optional attention mask [batch_size, sequence_length]
            normalize: Whether to normalize the output embeddings
            output_dim: Optional output dimension (must be ≤ config.output_dim)

        Returns:
            Embeddings [batch_size, output_dim]
        """
        # Generate token embeddings using transformer
        token_embeddings = self.transformer(input_ids)

        # Pool token embeddings
        pooled_embeddings = self.pooler(
            token_embeddings, attention_mask
        )

        # Project to target dimension
        embeddings = self.projection(pooled_embeddings)

        # Slice to requested dimension if specified
        if output_dim is not None:
            if output_dim > self.config.output_dim:
                raise ValueError(
                    f"Requested output_dim {output_dim} exceeds model's output_dim {self.config.output_dim}"
                )
            embeddings = embeddings[:, :output_dim]

        # Normalize embeddings if requested
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def save_pretrained(self, output_dir: Union[str, Path]) -> None:
        """
        Save model and configuration to directory.

        Args:
            output_dir: Directory to save model
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config_path = output_dir / "config.json"
        self.config.save_to_json(config_path)

        # Save model weights
        model_path = output_dir / "model.pt"
        torch.save(self.state_dict(), model_path)

        logger.info(f"Model saved to {output_dir}")

    @classmethod
    def from_pretrained(
        cls, model_dir: Union[str, Path]
    ) -> "GeminiEmbedding":
        """
        Load model and configuration from directory.

        Args:
            model_dir: Directory containing saved model

        Returns:
            Loaded model
        """
        model_dir = Path(model_dir)

        # Load configuration
        config_path = model_dir / "config.json"
        config = ModelConfig.from_json(config_path)

        # Create model
        model = cls(config)

        # Load weights
        model_path = model_dir / "model.pt"
        model.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )

        logger.info(f"Model loaded from {model_dir}")
        return model


class NoiseContrastiveLoss(nn.Module):
    """
    Noise-contrastive estimation (NCE) loss with in-batch negatives as described in the paper.

    The loss is designed for training embeddings with queries, positive targets, and optionally hard negatives.
    It also supports multi-resolution loss for different embedding dimensions.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize NCE loss.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.temperature = config.temperature
        self.mrl_dims = config.mrl_dims
        logger.info(
            f"Initialized NoiseContrastiveLoss with temperature={self.temperature}, mrl_dims={self.mrl_dims}"
        )

    def _compute_single_loss(
        self,
        query_embeddings: Tensor,
        positive_embeddings: Tensor,
        negative_embeddings: Optional[Tensor] = None,
        query_ids: Optional[Tensor] = None,
        positive_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute NCE loss for a single embedding dimension.

        Implementation of Equation (2) from the paper.

        Args:
            query_embeddings: Query embeddings [batch_size, embed_dim]
            positive_embeddings: Positive target embeddings [batch_size, embed_dim]
            negative_embeddings: Optional hard negative embeddings [batch_size, embed_dim]
            query_ids: Optional query IDs for masking [batch_size]
            positive_ids: Optional positive target IDs for masking [batch_size]

        Returns:
            NCE loss
        """
        batch_size = query_embeddings.size(0)

        # Compute query-positive similarities
        sim_q_pos = (
            torch.bmm(
                query_embeddings.view(batch_size, 1, -1),
                positive_embeddings.view(batch_size, -1, 1),
            ).squeeze(-1)
            / self.temperature
        )

        # Compute all query-positive similarities for in-batch negatives
        sim_q_all_pos = (
            torch.mm(
                query_embeddings, positive_embeddings.transpose(0, 1)
            )
            / self.temperature
        )

        # Create mask for in-batch negatives
        # Default mask allows all in-batch negatives
        mask = torch.ones_like(sim_q_all_pos)

        # Apply masking for same queries or same positives
        if query_ids is not None and positive_ids is not None:
            # Mask out where query_i = query_j or positive_i = positive_j
            same_q_mask = query_ids.unsqueeze(
                1
            ) == query_ids.unsqueeze(0)
            same_pos_mask = positive_ids.unsqueeze(
                1
            ) == positive_ids.unsqueeze(0)

            # Implementation of Equation (3) from the paper
            mask = mask * (~(same_q_mask | same_pos_mask)).float()

        # Always mask out diagonal (self-comparison)
        mask.fill_diagonal_(0)

        # Apply mask to similarities
        sim_q_all_pos = sim_q_all_pos * mask

        # If hard negatives are provided, compute similarities
        if negative_embeddings is not None:
            sim_q_neg = (
                torch.bmm(
                    query_embeddings.view(batch_size, 1, -1),
                    negative_embeddings.view(batch_size, -1, 1),
                ).squeeze(-1)
                / self.temperature
            )

            # Create denominator with hard negative and in-batch negatives
            # Fill masked positions with large negative value to zero them out in softmax
            sim_q_all_pos = sim_q_all_pos.masked_fill(
                mask == 0, -9e15
            )

            # Add query-negative similarities to denominator
            denominator = torch.cat(
                [sim_q_neg.view(-1, 1), sim_q_all_pos], dim=1
            )

            # Log softmax for numerical stability
            log_prob = F.log_softmax(denominator, dim=1)

            # First column contains the hard negative
            loss = -log_prob[:, 0].mean()
        else:
            # Only use in-batch negatives in denominator
            # Fill masked positions with large negative value to zero them out in softmax
            sim_q_all_pos = sim_q_all_pos.masked_fill(
                mask == 0, -9e15
            )

            # Log softmax for numerical stability
            log_prob = F.log_softmax(
                torch.cat(
                    [sim_q_pos.view(-1, 1), sim_q_all_pos], dim=1
                ),
                dim=1,
            )

            # First column contains the positive
            loss = -log_prob[:, 0].mean()

        return loss

    def forward(
        self,
        query_embeddings: Tensor,
        positive_embeddings: Tensor,
        negative_embeddings: Optional[Tensor] = None,
        query_ids: Optional[Tensor] = None,
        positive_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute multi-resolution NCE loss for different embedding dimensions.

        Args:
            query_embeddings: Query embeddings [batch_size, embed_dim]
            positive_embeddings: Positive target embeddings [batch_size, embed_dim]
            negative_embeddings: Optional hard negative embeddings [batch_size, embed_dim]
            query_ids: Optional query IDs for masking [batch_size]
            positive_ids: Optional positive target IDs for masking [batch_size]

        Returns:
            Dictionary with losses for each dimension and average loss
        """
        losses = {}
        avg_loss = 0.0

        for dim in self.mrl_dims:
            # Slice embeddings to current dimension
            q_emb = query_embeddings[:, :dim]
            pos_emb = positive_embeddings[:, :dim]
            neg_emb = (
                negative_embeddings[:, :dim]
                if negative_embeddings is not None
                else None
            )

            # Normalize embeddings for cosine similarity
            q_emb = F.normalize(q_emb, p=2, dim=1)
            pos_emb = F.normalize(pos_emb, p=2, dim=1)
            if neg_emb is not None:
                neg_emb = F.normalize(neg_emb, p=2, dim=1)

            # Compute loss for current dimension
            loss = self._compute_single_loss(
                q_emb, pos_emb, neg_emb, query_ids, positive_ids
            )
            losses[f"loss_{dim}"] = loss
            avg_loss += loss

        # Average loss across all dimensions
        avg_loss /= len(self.mrl_dims)
        losses["loss"] = avg_loss

        return losses


class CustomBatch:
    """
    A custom batch class for handling batches of data during training and evaluation.

    This class serves as a replacement for a full Dataset implementation, allowing
    direct feeding of tensor data into the model.
    """

    def __init__(
        self,
        query_input_ids: Tensor,
        query_attention_mask: Tensor,
        positive_input_ids: Tensor,
        positive_attention_mask: Tensor,
        negative_input_ids: Optional[Tensor] = None,
        negative_attention_mask: Optional[Tensor] = None,
        query_ids: Optional[Tensor] = None,
        positive_ids: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize batch.

        Args:
            query_input_ids: Input IDs for queries [batch_size, seq_length]
            query_attention_mask: Attention masks for queries [batch_size, seq_length]
            positive_input_ids: Input IDs for positive examples [batch_size, seq_length]
            positive_attention_mask: Attention masks for positive examples [batch_size, seq_length]
            negative_input_ids: Optional input IDs for negative examples [batch_size, seq_length]
            negative_attention_mask: Optional attention masks for negative examples [batch_size, seq_length]
            query_ids: Optional unique IDs for queries [batch_size]
            positive_ids: Optional unique IDs for positive examples [batch_size]
            device: Optional device to move tensors to
        """
        self.query_input_ids = query_input_ids
        self.query_attention_mask = query_attention_mask
        self.positive_input_ids = positive_input_ids
        self.positive_attention_mask = positive_attention_mask
        self.negative_input_ids = negative_input_ids
        self.negative_attention_mask = negative_attention_mask
        self.query_ids = query_ids
        self.positive_ids = positive_ids

        if device is not None:
            self.to(device)

    def to(self, device: torch.device) -> "CustomBatch":
        """
        Move all tensors in batch to specified device.

        Args:
            device: Device to move tensors to

        Returns:
            Self with tensors moved to device
        """
        self.query_input_ids = self.query_input_ids.to(device)
        self.query_attention_mask = self.query_attention_mask.to(
            device
        )
        self.positive_input_ids = self.positive_input_ids.to(device)
        self.positive_attention_mask = (
            self.positive_attention_mask.to(device)
        )

        if self.negative_input_ids is not None:
            self.negative_input_ids = self.negative_input_ids.to(
                device
            )
        if self.negative_attention_mask is not None:
            self.negative_attention_mask = (
                self.negative_attention_mask.to(device)
            )
        if self.query_ids is not None:
            self.query_ids = self.query_ids.to(device)
        if self.positive_ids is not None:
            self.positive_ids = self.positive_ids.to(device)

        return self

    def __getitem__(self, key: str) -> Optional[Tensor]:
        """
        Get item by key.

        Args:
            key: Key to get item for

        Returns:
            Tensor for key or None if key not found
        """
        return getattr(self, key, None)

    def items(self) -> List[Tuple[str, Optional[Tensor]]]:
        """
        Get all items in batch.

        Returns:
            List of (key, tensor) tuples
        """
        return [
            (name, getattr(self, name))
            for name in [
                "query_input_ids",
                "query_attention_mask",
                "positive_input_ids",
                "positive_attention_mask",
                "negative_input_ids",
                "negative_attention_mask",
                "query_ids",
                "positive_ids",
            ]
            if getattr(self, name) is not None
        ]

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get item by key with default.

        Args:
            key: Key to get item for
            default: Default value to return if key not found

        Returns:
            Tensor for key or default if key not found
        """
        return getattr(self, key, default)


def prepare_example_batch(
    tokenizer: Any,
    queries: List[str],
    positives: List[str],
    negatives: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    max_length: int = 512,
    device: Optional[torch.device] = None,
) -> CustomBatch:
    """
    Prepare a batch of examples for the model.

    Args:
        tokenizer: Tokenizer for encoding texts
        queries: List of query texts
        positives: List of positive target texts
        negatives: Optional list of hard negative target texts
        tasks: Optional list of task strings
        max_length: Maximum sequence length for tokenization
        device: Optional device to move tensors to

    Returns:
        Batch of examples
    """
    # Use default task if not provided
    if tasks is None:
        tasks = ["general"] * len(queries)

    # Prepare queries with task prefix
    query_texts = [
        f"{task}: {query}" if task else query
        for task, query in zip(tasks, queries)
    ]

    # Tokenize inputs
    query_encodings = tokenizer(
        query_texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    positive_encodings = tokenizer(
        positives,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Prepare batch
    batch_args = {
        "query_input_ids": query_encodings["input_ids"],
        "query_attention_mask": query_encodings["attention_mask"],
        "positive_input_ids": positive_encodings["input_ids"],
        "positive_attention_mask": positive_encodings[
            "attention_mask"
        ],
        "query_ids": torch.arange(len(queries)),
        "positive_ids": torch.arange(len(positives)),
    }

    # Add negatives if provided
    if negatives is not None:
        negative_encodings = tokenizer(
            negatives,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        batch_args.update(
            {
                "negative_input_ids": negative_encodings["input_ids"],
                "negative_attention_mask": negative_encodings[
                    "attention_mask"
                ],
            }
        )

    # Create and return batch
    return CustomBatch(**batch_args, device=device)


class GeminiEmbeddingTrainer:
    """
    Trainer for Gemini Embedding model.

    Handles the training loop, evaluation, and saving of the model.
    """

    def __init__(
        self,
        model: GeminiEmbedding,
        tokenizer: Any,
        train_queries: List[str],
        train_positives: List[str],
        train_negatives: Optional[List[str]] = None,
        train_tasks: Optional[List[str]] = None,
        eval_queries: Optional[List[str]] = None,
        eval_positives: Optional[List[str]] = None,
        eval_negatives: Optional[List[str]] = None,
        eval_tasks: Optional[List[str]] = None,
        config: Optional[ModelConfig] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Gemini Embedding model
            tokenizer: Tokenizer for encoding texts
            train_queries: List of training query texts
            train_positives: List of training positive target texts
            train_negatives: Optional list of training hard negative target texts
            train_tasks: Optional list of training task strings
            eval_queries: Optional list of evaluation query texts
            eval_positives: Optional list of evaluation positive target texts
            eval_negatives: Optional list of evaluation hard negative target texts
            eval_tasks: Optional list of evaluation task strings
            config: Optional model configuration (taken from model if not provided)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or model.config

        # Store training data
        self.train_queries = train_queries
        self.train_positives = train_positives
        self.train_negatives = train_negatives
        self.train_tasks = train_tasks

        # Store evaluation data
        self.eval_queries = eval_queries
        self.eval_positives = eval_positives
        self.eval_negatives = eval_negatives
        self.eval_tasks = eval_tasks

        # Set up optimizer

    def train(
        self,
        output_dir: Union[str, Path],
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        eval_steps: Optional[int] = None,
        save_steps: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            output_dir: Directory to save model
            num_epochs: Number of training epochs (defaults to config.num_train_epochs)
            batch_size: Batch size (defaults to config.batch_size)
            eval_steps: Evaluate every this many steps (defaults to once per epoch)
            save_steps: Save model every this many steps (defaults to once per epoch)

        Returns:
            Dictionary with training metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        num_epochs = num_epochs or self.config.num_train_epochs
        batch_size = batch_size or self.config.batch_size

        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        if self.eval_dataset:
            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
        else:
            eval_loader = None

        # Set default evaluation and saving steps
        steps_per_epoch = len(train_loader)
        eval_steps = eval_steps or steps_per_epoch
        save_steps = save_steps or steps_per_epoch

        # Initialize metrics
        metrics = {
            "train_loss": [],
            "eval_loss": [] if eval_loader else None,
        }

        device = next(self.model.parameters()).device
        logger.info(f"Training on device: {device}")

        # Training loop
        global_step = 0

        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
            self.model.train()
            epoch_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {
                    k: (
                        v.to(device)
                        if isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in batch.items()
                }

                # Forward pass
                query_embeddings = self.model(
                    batch["query_input_ids"],
                    batch.get("query_attention_mask"),
                    normalize=False,
                )

                positive_embeddings = self.model(
                    batch["positive_input_ids"],
                    batch.get("positive_attention_mask"),
                    normalize=False,
                )

                negative_embeddings = None
                if "negative_input_ids" in batch:
                    negative_embeddings = self.model(
                        batch["negative_input_ids"],
                        batch.get("negative_attention_mask"),
                        normalize=False,
                    )

                # Compute loss
                loss_dict = self.loss_fn(
                    query_embeddings,
                    positive_embeddings,
                    negative_embeddings,
                    batch.get("idx"),
                    batch.get("idx"),
                )

                loss = loss_dict["loss"]

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Clip gradients
                if self.config.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                # Update parameters
                self.optimizer.step()

                # Update metrics
                epoch_loss += loss.item()
                metrics["train_loss"].append(loss.item())

                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item()})

                global_step += 1

                # Evaluate if needed
                if eval_loader and global_step % eval_steps == 0:
                    eval_loss = self.evaluate(eval_loader)
                    metrics["eval_loss"].append(eval_loss)
                    logger.info(
                        f"Step {global_step}: eval_loss = {eval_loss:.4f}"
                    )

                # Save if needed
                if global_step % save_steps == 0:
                    save_path = (
                        output_dir / f"checkpoint-{global_step}"
                    )
                    self.model.save_pretrained(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

            # End of epoch
            epoch_loss /= len(train_loader)
            logger.info(
                f"Epoch {epoch+1} completed. Average loss: {epoch_loss:.4f}"
            )

            # Save at end of epoch
            save_path = output_dir / f"checkpoint-epoch-{epoch+1}"
            self.model.save_pretrained(save_path)
            logger.info(f"Saved checkpoint to {save_path}")

        # Save final model
        self.model.save_pretrained(output_dir)
        logger.info(
            f"Training completed. Final model saved to {output_dir}"
        )

        return metrics

    def evaluate(
        self, eval_loader: Optional[DataLoader] = None
    ) -> float:
        """
        Evaluate the model.

        Args:
            eval_loader: Optional evaluation data loader

        Returns:
            Average evaluation loss
        """
        if eval_loader is None:
            if self.eval_dataset is None:
                logger.warning("No evaluation dataset provided")
                return 0.0

            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

        self.model.eval()
        device = next(self.model.parameters()).device

        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                # Move batch to device
                batch = {
                    k: (
                        v.to(device)
                        if isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in batch.items()
                }

                # Forward pass
                query_embeddings = self.model(
                    batch["query_input_ids"],
                    batch.get("query_attention_mask"),
                    normalize=False,
                )

                positive_embeddings = self.model(
                    batch["positive_input_ids"],
                    batch.get("positive_attention_mask"),
                    normalize=False,
                )

                negative_embeddings = None
                if "negative_input_ids" in batch:
                    negative_embeddings = self.model(
                        batch["negative_input_ids"],
                        batch.get("negative_attention_mask"),
                        normalize=False,
                    )

                # Compute loss
                loss_dict = self.loss_fn(
                    query_embeddings,
                    positive_embeddings,
                    negative_embeddings,
                    batch.get("idx"),
                    batch.get("idx"),
                )

                loss = loss_dict["loss"]
                total_loss += loss.item()

        total_loss / len(eval_loader)
        self.model.train()


# model = create_bidirectional_transformer(ModelConfig())
# x = model.forward(torch.randint(0, 100, (1, 10)))
# print(x.shape)
