# Gemini Embedding Model Implementation

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
