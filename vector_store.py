import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import VECTOR_STORE_PATH, EMBEDDING_MODEL, EMBEDDING_DIMENSION, logger


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.embedding_model = None
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        self._load_embedding_model()

    def _load_embedding_model(self):
        """Load local embedding model"""
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"Model loaded successfully. Dimension: {EMBEDDING_DIMENSION}")

            # Test the model
            test_embedding = self.embedding_model.encode(["test"])
            actual_dim = test_embedding.shape[1]
            logger.info(f"Actual embedding dimension: {actual_dim}")

            if actual_dim != EMBEDDING_DIMENSION:
                logger.warning(f"Dimension mismatch! Expected {EMBEDDING_DIMENSION}, got {actual_dim}")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def generate_embeddings(self, texts):
        """Generate embeddings using local model"""
        if not self.embedding_model:
            raise ValueError("Embedding model not loaded")

        logger.info(f"Generating embeddings for {len(texts)} texts")

        # Generate embeddings in batches for memory efficiency
        batch_size = 32
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True  # Important for cosine similarity
            )
            embeddings.append(batch_embeddings)

        # Concatenate all batches
        all_embeddings = np.vstack(embeddings) if embeddings else np.empty((0, EMBEDDING_DIMENSION))

        logger.info(f"Generated embeddings shape: {all_embeddings.shape}")
        return all_embeddings.astype(np.float32)

    def build_index(self, chunks):
        """Build FAISS index from chunks"""
        logger.info(f"Building FAISS index for {len(chunks)} chunks")

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]

        # Generate embeddings
        embeddings = self.generate_embeddings(texts)

        # Create FAISS index (Inner Product for normalized embeddings = cosine similarity)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)

        # Add embeddings to index
        self.index.add(embeddings)

        logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors")

        # Log some statistics
        self._log_index_stats()

    def _log_index_stats(self):
        """Log index statistics"""
        if not self.chunks:
            return

        # Page distribution
        pages = [chunk["page_number"] for chunk in self.chunks]
        unique_pages = len(set(pages))

        # Token statistics
        token_counts = [chunk.get("token_count", 0) for chunk in self.chunks]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

        # Complete pages vs chunks
        complete_pages = sum(1 for chunk in self.chunks if chunk.get("is_complete_page", False))

        logger.info(f"Index Statistics:")
        logger.info(f"  - Total chunks: {len(self.chunks)}")
        logger.info(f"  - Pages covered: {unique_pages}")
        logger.info(f"  - Complete pages: {complete_pages}")
        logger.info(f"  - Chunked pages: {unique_pages - complete_pages}")
        logger.info(f"  - Average tokens per chunk: {avg_tokens:.1f}")

    def search(self, query, k=5):
        """Search for similar chunks"""
        if not self.index:
            logger.warning("No index available for search")
            return []

        # Generate query embedding
        query_embedding = self.generate_embeddings([query])

        # Search
        scores, indices = self.index.search(query_embedding, k)

        # Return results with chunks and scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append((chunk, float(score)))

        logger.info(f"Found {len(results)} results for query")
        return results

    def get_page_chunks(self, page_number):
        """Get all chunks from a specific page"""
        page_chunks = [chunk for chunk in self.chunks if chunk["page_number"] == page_number]
        return page_chunks

    def get_stats(self):
        """Get comprehensive statistics"""
        if not self.chunks:
            return {"total_chunks": 0, "pages": 0}

        pages = [chunk["page_number"] for chunk in self.chunks]
        token_counts = [chunk.get("token_count", 0) for chunk in self.chunks]

        return {
            "total_chunks": len(self.chunks),
            "unique_pages": len(set(pages)),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(token_counts) if token_counts else 0,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "model_name": EMBEDDING_MODEL,
            "complete_pages": sum(1 for chunk in self.chunks if chunk.get("is_complete_page", False))
        }