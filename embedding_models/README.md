# Embedding Models Directory

This directory contains the embedding model weights used for CLIP-CC-Bench evaluation.

## Directory Structure

```
embedding_models/
├── NV-Embed-v2/
├── KaLM-Embedding-Gemma3-12B-2511/
├── llama-embed-nemotron-8b/
├── gte-Qwen2-7B-instruct/
└── Qwen3-Embedding-8B/
```

## Usage

Place the downloaded embedding model files directly in their respective directories. The evaluation scripts will automatically detect and load models from these locations.

**Note**: Model weights are NOT tracked in git. Only the directory structure is version controlled.
