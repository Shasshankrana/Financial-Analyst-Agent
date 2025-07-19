import os
import time
import psutil
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, logger


class VectorStore:
    def __init__(self):
        logger.info("Initializing VectorStore...")
        start_time = time.time()

        self.index = None
        self.chunks = []
        self.embedding_model = None

        try:
            # Load embedding model
            self._load_embedding_model()

            init_time = time.time() - start_time
            logger.info(f"VectorStore initialized successfully in {init_time:.2f}s")

        except Exception as e:
            init_time = time.time() - start_time
            logger.error(f"VectorStore initialization failed after {init_time:.2f}s: {e}", exc_info=True)
            raise

    def _load_embedding_model(self):
        """Load local embedding model"""
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        start_time = time.time()

        # Log system resources before loading
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        logger.debug(f"Memory usage before model loading: {memory_before:.1f}MB")

        try:
            logger.debug("Downloading/loading model from cache...")
            model_start = time.time()

            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

            model_time = time.time() - model_start
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before

            logger.info(f"Model loaded successfully:")
            logger.info(f"  - Loading time: {model_time:.2f}s")
            logger.info(f"  - Memory usage: +{memory_used:.1f}MB (total: {memory_after:.1f}MB)")
            logger.info(f"  - Target dimension: {EMBEDDING_DIMENSION}")

            # Test the model with actual embedding
            logger.debug("Testing model with sample embedding...")
            test_start = time.time()
            test_embedding = self.embedding_model.encode(["test sentence for dimension validation"])
            test_time = time.time() - test_start

            actual_dim = test_embedding.shape[1]
            logger.info(f"Model validation:")
            logger.info(f"  - Actual dimension: {actual_dim}")
            logger.info(f"  - Test embedding time: {test_time:.3f}s")
            logger.info(f"  - Embedding shape: {test_embedding.shape}")
            logger.info(f"  - Embedding dtype: {test_embedding.dtype}")

            if actual_dim != EMBEDDING_DIMENSION:
                logger.error(f"CRITICAL: Dimension mismatch! Expected {EMBEDDING_DIMENSION}, got {actual_dim}")
                raise ValueError(f"Model dimension mismatch: expected {EMBEDDING_DIMENSION}, got {actual_dim}")

            # Log model details
            if hasattr(self.embedding_model, 'max_seq_length'):
                logger.info(f"  - Max sequence length: {self.embedding_model.max_seq_length}")

            total_time = time.time() - start_time
            logger.info(f"Embedding model setup completed in {total_time:.2f}s")

        except Exception as e:
            load_time = time.time() - start_time
            memory_current = process.memory_info().rss / 1024 / 1024
            logger.error(f"Failed to load embedding model after {load_time:.2f}s", exc_info=True)
            logger.error(f"Memory at failure: {memory_current:.1f}MB")
            raise

    def generate_embeddings(self, texts):
        """Generate embeddings using local model"""
        logger.info(f"Starting embedding generation for {len(texts)} texts")
        start_time = time.time()

        if not self.embedding_model:
            logger.error("Embedding model not loaded")
            raise ValueError("Embedding model not loaded")

        if not texts:
            logger.warning("No texts provided for embedding generation")
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

        try:
            # Log text statistics
            text_lengths = [len(text) for text in texts]
            avg_length = sum(text_lengths) / len(text_lengths)
            min_length = min(text_lengths)
            max_length = max(text_lengths)

            logger.info(f"Text statistics:")
            logger.info(f"  - Count: {len(texts)}")
            logger.info(f"  - Average length: {avg_length:.0f} characters")
            logger.info(f"  - Length range: {min_length} - {max_length} characters")
            logger.info(f"  - Total characters: {sum(text_lengths):,}")

            # Generate embeddings in batches for memory efficiency
            batch_size = 32
            embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size

            logger.info(f"Processing in {total_batches} batches of {batch_size}")

            for i in range(0, len(texts), batch_size):
                batch_start = time.time()
                batch_num = i // batch_size + 1

                batch = texts[i:i + batch_size]
                batch_size_actual = len(batch)

                logger.debug(f"Processing batch {batch_num}/{total_batches} ({batch_size_actual} texts)")

                # Generate embeddings for batch
                encoding_start = time.time()
                batch_embeddings = self.embedding_model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,  # Important for cosine similarity
                    show_progress_bar=False  # We handle progress logging
                )
                encoding_time = time.time() - encoding_start

                embeddings.append(batch_embeddings)

                batch_time = time.time() - batch_start
                texts_per_sec = batch_size_actual / batch_time

                logger.debug(f"Batch {batch_num} completed:")
                logger.debug(f"  - Encoding time: {encoding_time:.3f}s")
                logger.debug(f"  - Total time: {batch_time:.3f}s")
                logger.debug(f"  - Speed: {texts_per_sec:.1f} texts/sec")

                # Progress update for large batches
                if total_batches > 10 and batch_num % max(1, total_batches // 10) == 0:
                    progress = (batch_num / total_batches) * 100
                    logger.info(f"Embedding progress: {progress:.1f}% ({batch_num}/{total_batches} batches)")

            # Concatenate all batches
            logger.debug("Concatenating batch embeddings...")
            concat_start = time.time()

            if embeddings:
                all_embeddings = np.vstack(embeddings)
            else:
                all_embeddings = np.empty((0, EMBEDDING_DIMENSION))

            concat_time = time.time() - concat_start

            # Convert to float32 for FAISS
            logger.debug("Converting to float32...")
            convert_start = time.time()
            all_embeddings = all_embeddings.astype(np.float32)
            convert_time = time.time() - convert_start

            total_time = time.time() - start_time
            texts_per_sec_total = len(texts) / total_time if total_time > 0 else 0

            # Log final statistics
            logger.info(f"Embedding generation completed:")
            logger.info(f"  - Shape: {all_embeddings.shape}")
            logger.info(f"  - Data type: {all_embeddings.dtype}")
            logger.info(f"  - Total time: {total_time:.2f}s")
            logger.info(f"  - Concatenation time: {concat_time:.3f}s")
            logger.info(f"  - Conversion time: {convert_time:.3f}s")
            logger.info(f"  - Overall speed: {texts_per_sec_total:.1f} texts/sec")
            logger.info(f"  - Memory size: {all_embeddings.nbytes / 1024 / 1024:.1f}MB")

            # Validate embeddings
            if len(all_embeddings) != len(texts):
                logger.error(f"Embedding count mismatch: {len(all_embeddings)} vs {len(texts)} texts")
                raise ValueError("Embedding generation failed: count mismatch")

            # Check for any invalid embeddings
            invalid_count = np.isnan(all_embeddings).sum()
            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} NaN values in embeddings")

            zero_embeddings = np.all(all_embeddings == 0, axis=1).sum()
            if zero_embeddings > 0:
                logger.warning(f"Found {zero_embeddings} zero embeddings (might indicate issues)")

            return all_embeddings

        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"Embedding generation failed after {generation_time:.2f}s: {e}", exc_info=True)
            raise

    def build_index(self, chunks):
        """Build FAISS index from chunks"""
        logger.info(f"Starting FAISS index construction for {len(chunks)} chunks")
        start_time = time.time()

        if not chunks:
            logger.warning("No chunks provided for index building")
            return

        try:
            self.chunks = chunks
            texts = [chunk["text"] for chunk in chunks]

            logger.info(f"Extracting text from {len(chunks)} chunks...")

            # Log chunk statistics
            self._log_chunk_stats(chunks)

            # Generate embeddings
            logger.info("Generating embeddings for all chunks...")
            embedding_start = time.time()
            embeddings = self.generate_embeddings(texts)
            embedding_time = time.time() - embedding_start

            logger.info(f"Embeddings generated in {embedding_time:.2f}s")

            # Create FAISS index
            logger.info(f"Creating FAISS index (Inner Product for cosine similarity)")
            index_start = time.time()

            self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            logger.debug(f"FAISS index created: {type(self.index).__name__}")

            # Add embeddings to index
            logger.info("Adding embeddings to FAISS index...")
            add_start = time.time()
            self.index.add(embeddings)
            add_time = time.time() - add_start

            index_build_time = time.time() - index_start
            total_time = time.time() - start_time

            logger.info(f"FAISS index built successfully:")
            logger.info(f"  - Vectors in index: {self.index.ntotal}")
            logger.info(f"  - Index dimension: {self.index.d}")
            logger.info(f"  - Index type: {type(self.index).__name__}")
            logger.info(f"  - Embedding time: {embedding_time:.2f}s")
            logger.info(f"  - Index creation time: {index_build_time:.2f}s")
            logger.info(f"  - Vector addition time: {add_time:.2f}s")
            logger.info(f"  - Total time: {total_time:.2f}s")

            # Validate index
            if self.index.ntotal != len(chunks):
                logger.error(f"Index validation failed: {self.index.ntotal} vectors vs {len(chunks)} chunks")
                raise ValueError("Index building failed: vector count mismatch")

            # Log detailed statistics
            self._log_index_stats()

            # Test search functionality
            self._test_index_functionality()

        except Exception as e:
            build_time = time.time() - start_time
            logger.error(f"FAISS index building failed after {build_time:.2f}s: {e}", exc_info=True)
            raise

    def _log_chunk_stats(self, chunks):
        """Log detailed chunk statistics"""
        logger.debug("Analyzing chunk statistics...")

        try:
            # Page distribution
            pages = [chunk["page_number"] for chunk in chunks]
            unique_pages = sorted(set(pages))
            page_counts = {page: pages.count(page) for page in unique_pages}

            # Token statistics
            token_counts = [chunk.get("token_count", 0) for chunk in chunks]

            # Text length statistics
            text_lengths = [len(chunk["text"]) for chunk in chunks]

            # Complete pages vs chunks
            complete_pages = [chunk for chunk in chunks if chunk.get("is_complete_page", False)]
            chunked_pages = [chunk for chunk in chunks if not chunk.get("is_complete_page", False)]

            logger.info(f"Chunk analysis:")
            logger.info(f"  - Total chunks: {len(chunks)}")
            logger.info(f"  - Pages covered: {len(unique_pages)} (pages {min(unique_pages)}-{max(unique_pages)})")
            logger.info(f"  - Complete pages: {len(complete_pages)}")
            logger.info(f"  - Chunked content: {len(chunked_pages)} chunks")

            if token_counts:
                avg_tokens = sum(token_counts) / len(token_counts)
                min_tokens = min(token_counts)
                max_tokens = max(token_counts)
                logger.info(f"  - Token stats: avg={avg_tokens:.0f}, min={min_tokens}, max={max_tokens}")

            if text_lengths:
                avg_chars = sum(text_lengths) / len(text_lengths)
                min_chars = min(text_lengths)
                max_chars = max(text_lengths)
                logger.info(f"  - Character stats: avg={avg_chars:.0f}, min={min_chars}, max={max_chars}")

            # Log pages with multiple chunks
            multi_chunk_pages = {page: count for page, count in page_counts.items() if count > 1}
            if multi_chunk_pages:
                logger.debug(f"Pages with multiple chunks: {len(multi_chunk_pages)}")
                for page, count in sorted(multi_chunk_pages.items())[:5]:  # Show first 5
                    logger.debug(f"  - Page {page}: {count} chunks")
                if len(multi_chunk_pages) > 5:
                    logger.debug(f"  ... and {len(multi_chunk_pages) - 5} more pages")

        except Exception as e:
            logger.error(f"Failed to analyze chunk statistics: {e}")

    def _log_index_stats(self):
        """Log comprehensive index statistics"""
        logger.debug("Generating index statistics...")

        try:
            if not self.chunks:
                logger.warning("No chunks available for statistics")
                return

            # Page distribution
            pages = [chunk["page_number"] for chunk in self.chunks]
            unique_pages = len(set(pages))

            # Token statistics
            token_counts = [chunk.get("token_count", 0) for chunk in self.chunks]
            total_tokens = sum(token_counts)
            avg_tokens = total_tokens / len(token_counts) if token_counts else 0

            # Complete pages vs chunks
            complete_pages = sum(1 for chunk in self.chunks if chunk.get("is_complete_page", False))
            chunked_pages = unique_pages - complete_pages

            # Index memory usage
            index_memory = 0
            if self.index:
                # Rough estimate: ntotal * dimension * 4 bytes (float32)
                index_memory = self.index.ntotal * self.index.d * 4 / 1024 / 1024  # MB

            logger.info(f"Vector Store Statistics:")
            logger.info(f"  - Total chunks: {len(self.chunks)}")
            logger.info(f"  - Pages covered: {unique_pages}")
            logger.info(f"  - Complete pages: {complete_pages}")
            logger.info(f"  - Chunked pages: {chunked_pages}")
            logger.info(f"  - Total tokens: {total_tokens:,}")
            logger.info(f"  - Average tokens per chunk: {avg_tokens:.1f}")
            logger.info(f"  - Index memory usage: ~{index_memory:.1f}MB")
            logger.info(f"  - Embedding dimension: {EMBEDDING_DIMENSION}")
            logger.info(f"  - Model: {EMBEDDING_MODEL}")

        except Exception as e:
            logger.error(f"Failed to generate index statistics: {e}")

    def _test_index_functionality(self):
        """Test basic index functionality"""
        logger.debug("Testing index functionality...")

        try:
            if not self.index or not self.chunks:
                logger.warning("Cannot test index: index or chunks not available")
                return

            # Test with a sample query
            test_query = "financial performance revenue"
            test_start = time.time()

            query_embedding = self.generate_embeddings([test_query])
            scores, indices = self.index.search(query_embedding, k=min(3, len(self.chunks)))

            test_time = time.time() - test_start

            # Validate results
            valid_results = sum(1 for idx in indices[0] if 0 <= idx < len(self.chunks))

            logger.info(f"Index functionality test:")
            logger.info(f"  - Test query: '{test_query}'")
            logger.info(f"  - Search time: {test_time:.3f}s")
            logger.info(f"  - Results returned: {len(indices[0])}")
            logger.info(f"  - Valid results: {valid_results}")

            if valid_results > 0:
                best_score = scores[0][0] if len(scores[0]) > 0 else 0
                logger.info(f"  - Best match score: {best_score:.4f}")
                logger.info("✅ Index test passed")
            else:
                logger.warning("⚠️ Index test returned no valid results")

        except Exception as e:
            logger.error(f"Index functionality test failed: {e}")

    def search(self, query, k=5):
        """Search for similar chunks"""
        logger.debug(f"Searching for query: '{query[:50]}...' (k={k})")
        start_time = time.time()

        if not self.index:
            logger.warning("No index available for search")
            return []

        if not query.strip():
            logger.warning("Empty query provided")
            return []

        try:
            # Generate query embedding
            embedding_start = time.time()
            query_embedding = self.generate_embeddings([query])
            embedding_time = time.time() - embedding_start

            # Perform search
            search_start = time.time()
            scores, indices = self.index.search(query_embedding, k)
            search_time = time.time() - search_start

            # Process results
            process_start = time.time()
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    results.append((chunk, float(score)))
                else:
                    logger.warning(f"Invalid index returned: {idx} (max: {len(self.chunks) - 1})")

            process_time = time.time() - process_start
            total_time = time.time() - start_time

            # Log search performance
            logger.info(f"Search completed:")
            logger.info(f"  - Query length: {len(query)} characters")
            logger.info(f"  - Embedding time: {embedding_time:.3f}s")
            logger.info(f"  - Search time: {search_time:.3f}s")
            logger.info(f"  - Processing time: {process_time:.3f}s")
            logger.info(f"  - Total time: {total_time:.3f}s")
            logger.info(f"  - Results found: {len(results)}")

            # Log result quality
            if results:
                scores_list = [score for _, score in results]
                avg_score = sum(scores_list) / len(scores_list)
                best_score = max(scores_list)
                worst_score = min(scores_list)

                logger.info(f"  - Score range: {worst_score:.4f} - {best_score:.4f}")
                logger.info(f"  - Average score: {avg_score:.4f}")

                # Log top result details
                top_chunk, top_score = results[0]
                logger.debug(f"  - Top result: Page {top_chunk['page_number']}, "
                             f"Chunk {top_chunk.get('chunk_index', 0)}, Score: {top_score:.4f}")
            else:
                logger.warning("No results found for query")

            return results

        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"Search failed after {search_time:.3f}s for query '{query[:50]}...': {e}", exc_info=True)
            return []

    def get_page_chunks(self, page_number):
        """Get all chunks from a specific page"""
        logger.debug(f"Retrieving chunks for page {page_number}")

        try:
            page_chunks = [chunk for chunk in self.chunks if chunk["page_number"] == page_number]

            logger.debug(f"Found {len(page_chunks)} chunks for page {page_number}")

            if page_chunks:
                token_count = sum(chunk.get("token_count", 0) for chunk in page_chunks)
                logger.debug(f"Page {page_number} total tokens: {token_count}")

            return page_chunks

        except Exception as e:
            logger.error(f"Failed to retrieve chunks for page {page_number}: {e}")
            return []

    def get_stats(self):
        """Get comprehensive statistics"""
        logger.debug("Generating comprehensive statistics...")

        try:
            if not self.chunks:
                return {"total_chunks": 0, "pages": 0}

            pages = [chunk["page_number"] for chunk in self.chunks]
            token_counts = [chunk.get("token_count", 0) for chunk in self.chunks]

            stats = {
                "total_chunks": len(self.chunks),
                "unique_pages": len(set(pages)),
                "total_tokens": sum(token_counts),
                "avg_tokens_per_chunk": sum(token_counts) / len(token_counts) if token_counts else 0,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "model_name": EMBEDDING_MODEL,
                "complete_pages": sum(1 for chunk in self.chunks if chunk.get("is_complete_page", False)),
                "index_available": self.index is not None,
                "vectors_in_index": self.index.ntotal if self.index else 0
            }

            logger.debug(f"Statistics generated: {len(stats)} metrics")
            return stats

        except Exception as e:
            logger.error(f"Failed to generate statistics: {e}")
            return {"error": str(e)}