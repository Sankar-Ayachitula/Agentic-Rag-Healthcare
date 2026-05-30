"""Central configuration.

Loads settings from environment (.env) so no secrets live in source.
Will expose: OPENAI_API_KEY, OPENAI_MODEL, EMBEDDING_MODEL, CHROMA_DIR,
and dataset/artifact paths used across training and serving.
"""
