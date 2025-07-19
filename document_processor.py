import fitz
import io
import time
import asyncio
from PIL import Image
import google.generativeai as genai
from transformers import AutoTokenizer
from config import GEMINI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, OCR_PROMPT, logger
import re

genai.configure(api_key=GEMINI_API_KEY)


class DocumentProcessor:
    def __init__(self, max_concurrent_requests=10):
        logger.info("Initializing DocumentProcessor...")

        try:
            logger.debug("Loading Gemini vision model...")
            self.vision_model = genai.GenerativeModel("gemini-2.0-flash")
            logger.info("Gemini vision model loaded successfully")

            # Load tokenizer for accurate token counting
            logger.debug(f"Loading tokenizer: {EMBEDDING_MODEL}")
            start_time = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
            load_time = time.time() - start_time
            logger.info(f"Tokenizer loaded successfully in {load_time:.2f}s")

            # Initialize semaphore for concurrent requests
            self.semaphore = asyncio.Semaphore(max_concurrent_requests)
            self.max_concurrent_requests = max_concurrent_requests

            logger.info(f"DocumentProcessor initialized with:")
            logger.info(f"  - Embedding model: {EMBEDDING_MODEL}")
            logger.info(f"  - Chunk size: {CHUNK_SIZE} tokens")
            logger.info(f"  - Chunk overlap: {CHUNK_OVERLAP} tokens")
            logger.info(f"  - Max concurrent requests: {max_concurrent_requests}")

        except Exception as e:
            logger.error(f"Failed to initialize DocumentProcessor: {e}", exc_info=True)
            raise

    def extract_pages_as_images(self, pdf_path):
        """Extract each page as high-quality image"""
        logger.info(f"Starting PDF page extraction from: {pdf_path}")
        start_time = time.time()

        try:
            doc = fitz.open(pdf_path)
            logger.info(f"PDF opened successfully - {doc.page_count} pages found")

            if doc.page_count == 0:
                logger.warning("PDF contains no pages")
                return []

            pages = []

            for page_num in range(doc.page_count):
                page_start = time.time()
                logger.debug(f"Processing page {page_num + 1}/{doc.page_count}")

                page = doc[page_num]

                # Convert to high-resolution image (3x for better OCR)
                logger.debug(f"Converting page {page_num + 1} to high-res image (3x scale)")
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
                img_data = pix.tobytes("png")

                page_info = {
                    "page_number": page_num + 1,
                    "image_data": img_data,
                    "width": pix.width,
                    "height": pix.height,
                    "size_bytes": len(img_data)
                }

                pages.append(page_info)

                page_time = time.time() - page_start
                logger.debug(f"Page {page_num + 1} extracted: {pix.width}x{pix.height}px, "
                             f"{len(img_data) / 1024:.1f}KB in {page_time:.2f}s")

            doc.close()

            total_time = time.time() - start_time
            total_size = sum(p["size_bytes"] for p in pages) / 1024 / 1024  # MB

            logger.info(f"Page extraction completed:")
            logger.info(f"  - {len(pages)} pages extracted in {total_time:.2f}s")
            logger.info(f"  - Total image data: {total_size:.1f}MB")
            logger.info(f"  - Average time per page: {total_time / len(pages):.2f}s")

            return pages

        except Exception as e:
            logger.error(f"Failed to extract pages from PDF: {e}", exc_info=True)
            raise

    async def extract_comprehensive_content(self, image_data, page_number):
        """Use Gemini Vision to extract ALL content in natural language"""
        logger.debug(f"Starting content extraction for page {page_number}")
        start_time = time.time()

        try:
            # Prepare image
            logger.debug(f"Loading image data for page {page_number} ({len(image_data) / 1024:.1f}KB)")
            image = Image.open(io.BytesIO(image_data))
            logger.debug(f"Image loaded: {image.size[0]}x{image.size[1]}px, mode: {image.mode}")

            # Call Gemini Vision API
            logger.debug(f"Sending page {page_number} to Gemini Vision API...")
            api_start = time.time()
            response = await self.vision_model.generate_content_async([OCR_PROMPT, image])
            api_time = time.time() - api_start

            extracted_text = response.text
            total_time = time.time() - start_time

            # Log extraction results
            char_count = len(extracted_text)
            word_count = len(extracted_text.split())
            line_count = len(extracted_text.split('\n'))

            logger.info(f"Page {page_number} content extracted successfully:")
            logger.info(f"  - Characters: {char_count:,}")
            logger.info(f"  - Words: {word_count:,}")
            logger.info(f"  - Lines: {line_count:,}")
            logger.info(f"  - API time: {api_time:.2f}s")
            logger.info(f"  - Total time: {total_time:.2f}s")

            if char_count < 100:
                logger.warning(f"Page {page_number} extracted very little content ({char_count} chars) - "
                               "might be mostly images or blank")

            return extracted_text

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Vision processing failed for page {page_number} after {processing_time:.2f}s: {e}",
                         exc_info=True)

            # Return error placeholder but don't fail completely
            error_text = f"Error processing page {page_number}: Content could not be extracted due to: {str(e)}"
            logger.warning(f"Returning error placeholder for page {page_number}")
            return error_text

    async def _process_single_page_with_semaphore(self, page_data):
        """Process a single page with semaphore control for concurrency"""
        page_num = page_data["page_number"]

        async with self.semaphore:
            logger.debug(f"Acquired semaphore for page {page_num}")
            start_time = time.time()

            try:
                # Extract comprehensive content using vision
                content = await self.extract_comprehensive_content(
                    page_data["image_data"], page_num
                )

                processed_page = {
                    "page_number": page_num,
                    "content": content,
                    "content_length": len(content),
                    "image_dimensions": {
                        "width": page_data["width"],
                        "height": page_data["height"]
                    }
                }

                processing_time = time.time() - start_time
                logger.info(f"Page {page_num} completed successfully in {processing_time:.2f}s")

                return processed_page, None  # (result, error)

            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"Failed to process page {page_num} in {processing_time:.2f}s: {e}")
                return None, e  # (result, error)

            finally:
                logger.debug(f"Released semaphore for page {page_num}")

    async def process_document(self, pdf_path):
        """Process entire document - each page becomes one document with concurrent processing"""
        logger.info(f"Starting concurrent document processing: {pdf_path}")
        start_time = time.time()

        try:
            # Extract all pages as images
            logger.info("Phase 1: Extracting pages as images...")
            pages = self.extract_pages_as_images(pdf_path)

            if not pages:
                logger.error("No pages extracted from PDF")
                raise ValueError("No pages found in PDF")

            # Process all pages concurrently with vision model
            logger.info(f"Phase 2: Processing {len(pages)} pages concurrently with vision model...")
            logger.info(f"Using up to {self.max_concurrent_requests} concurrent requests")

            # Create tasks for all pages
            tasks = [
                self._process_single_page_with_semaphore(page_data)
                for page_data in pages
            ]

            # Execute all tasks concurrently
            logger.info(f"Starting concurrent processing of {len(tasks)} pages...")
            concurrent_start = time.time()

            results = await asyncio.gather(*tasks, return_exceptions=True)

            concurrent_time = time.time() - concurrent_start
            logger.info(f"Concurrent processing completed in {concurrent_time:.2f}s")

            # Process results
            processed_pages = []
            successful_pages = 0
            failed_pages = 0

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {i + 1} failed with exception: {result}")
                    failed_pages += 1
                    continue

                processed_page, error = result

                if error:
                    logger.error(f"Page {pages[i]['page_number']} failed: {error}")
                    failed_pages += 1
                elif processed_page:
                    processed_pages.append(processed_page)
                    successful_pages += 1

            # Sort processed pages by page number to maintain order
            processed_pages.sort(key=lambda x: x["page_number"])

            total_time = time.time() - start_time
            total_content_length = sum(p["content_length"] for p in processed_pages)

            # Calculate speedup
            estimated_sequential_time = concurrent_time * self.max_concurrent_requests
            speedup = estimated_sequential_time / concurrent_time if concurrent_time > 0 else 1

            logger.info(f"Document processing completed:")
            logger.info(f"  - Total time: {total_time:.2f}s")
            logger.info(f"  - Concurrent processing time: {concurrent_time:.2f}s")
            logger.info(f"  - Estimated speedup: {speedup:.1f}x")
            logger.info(f"  - Successful pages: {successful_pages}")
            logger.info(f"  - Failed pages: {failed_pages}")
            logger.info(f"  - Total content: {total_content_length:,} characters")

            if processed_pages:
                logger.info(f"  - Average content per page: {total_content_length / len(processed_pages):,.0f} chars")
                logger.info(f"  - Average time per page: {concurrent_time / len(pages):.2f}s")

            if failed_pages > 0:
                logger.warning(f"{failed_pages} pages failed to process - check logs for details")

            return processed_pages

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Document processing failed after {processing_time:.2f}s: {e}", exc_info=True)
            raise

    def count_tokens(self, text):
        """Count tokens using the actual tokenizer"""
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            token_count = len(tokens)
            logger.debug(f"Token count: {token_count} for {len(text)} characters")
            return token_count
        except Exception as e:
            logger.error(f"Failed to count tokens: {e}")
            # Fallback to approximate counting
            approx_tokens = len(text.split()) * 1.3  # rough approximation
            logger.warning(f"Using approximate token count: {approx_tokens}")
            return int(approx_tokens)

    def create_chunks_from_page(self, page_content, page_number):
        """Create overlapping chunks from a single page's content"""
        logger.debug(f"Starting chunk creation for page {page_number}")
        start_time = time.time()

        try:
            content = page_content["content"]
            content_length = len(content)

            logger.debug(f"Page {page_number} content: {content_length:,} characters")

            # Count total tokens
            total_tokens = self.count_tokens(content)
            logger.info(f"Page {page_number}: {total_tokens} tokens, target chunk size: {CHUNK_SIZE}")

            # If content is short enough, return as single chunk
            if total_tokens <= CHUNK_SIZE:
                chunk = {
                    "text": content,
                    "page_number": page_number,
                    "chunk_index": 0,
                    "chunk_id": f"page_{page_number}_chunk_0",
                    "token_count": total_tokens,
                    "is_complete_page": True
                }

                processing_time = time.time() - start_time
                logger.info(f"Page {page_number} fits in single chunk ({total_tokens} tokens) - "
                            f"completed in {processing_time:.3f}s")
                return [chunk]

            # Split into sentences for better chunking
            logger.debug(f"Splitting page {page_number} content into sentences...")
            sentences = self._split_into_sentences(content)
            logger.debug(f"Found {len(sentences)} sentences in page {page_number}")

            chunks = []
            current_chunk = ""
            current_tokens = 0
            chunk_index = 0

            for i, sentence in enumerate(sentences):
                sentence_tokens = self.count_tokens(sentence)

                # If adding this sentence would exceed chunk size, save current chunk
                if current_tokens + sentence_tokens > CHUNK_SIZE and current_chunk:
                    chunk = {
                        "text": current_chunk.strip(),
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "chunk_id": f"page_{page_number}_chunk_{chunk_index}",
                        "token_count": current_tokens,
                        "is_complete_page": False
                    }
                    chunks.append(chunk)

                    logger.debug(f"Created chunk {chunk_index} for page {page_number}: "
                                 f"{current_tokens} tokens, {len(current_chunk)} chars")

                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk, CHUNK_OVERLAP)
                    overlap_tokens = self.count_tokens(overlap_text)
                    logger.debug(f"Using overlap: {overlap_tokens} tokens")

                    current_chunk = overlap_text + " " + sentence
                    current_tokens = self.count_tokens(current_chunk)
                    chunk_index += 1
                else:
                    current_chunk += " " + sentence
                    current_tokens += sentence_tokens

                # Log progress for long pages
                if i > 0 and i % 50 == 0:
                    logger.debug(f"Processed {i}/{len(sentences)} sentences in page {page_number}")

            # Add final chunk if it has content
            if current_chunk.strip():
                final_chunk = {
                    "text": current_chunk.strip(),
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "chunk_id": f"page_{page_number}_chunk_{chunk_index}",
                    "token_count": current_tokens,
                    "is_complete_page": False
                }
                chunks.append(final_chunk)

                logger.debug(f"Created final chunk {chunk_index} for page {page_number}: "
                             f"{current_tokens} tokens")

            processing_time = time.time() - start_time
            avg_tokens_per_chunk = sum(c["token_count"] for c in chunks) / len(chunks)

            logger.info(f"Page {page_number} chunking completed:")
            logger.info(f"  - Created {len(chunks)} chunks in {processing_time:.3f}s")
            logger.info(f"  - Average tokens per chunk: {avg_tokens_per_chunk:.0f}")
            logger.info(f"  - Token efficiency: {(total_tokens / sum(c['token_count'] for c in chunks) * 100):.1f}%")

            return chunks

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Chunk creation failed for page {page_number} after {processing_time:.3f}s: {e}",
                         exc_info=True)
            raise

    def _split_into_sentences(self, text):
        """Split text into sentences for better chunking"""
        logger.debug("Splitting text into sentences...")
        try:
            # Simple sentence splitting - can be improved with spaCy if needed
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            logger.debug(f"Split into {len(sentences)} sentences")

            # Log statistics about sentence lengths
            if sentences:
                lengths = [len(s) for s in sentences]
                avg_length = sum(lengths) / len(lengths)
                min_length = min(lengths)
                max_length = max(lengths)

                logger.debug(f"Sentence length stats: avg={avg_length:.0f}, min={min_length}, max={max_length}")

            return sentences

        except Exception as e:
            logger.error(f"Failed to split text into sentences: {e}")
            # Fallback: split by newlines
            logger.warning("Falling back to newline-based splitting")
            return [line.strip() for line in text.split('\n') if line.strip()]

    def _get_overlap_text(self, text, overlap_tokens):
        """Get the last N tokens worth of text for overlap"""
        try:
            words = text.split()

            # Approximate: assume ~1.3 words per token (varies by tokenizer)
            overlap_words = int(overlap_tokens * 1.3)

            if len(words) <= overlap_words:
                logger.debug(f"Overlap uses entire text ({len(words)} words)")
                return text

            overlap_text = " ".join(words[-overlap_words:])
            logger.debug(f"Created overlap: {overlap_words} words from {len(words)} total words")
            return overlap_text

        except Exception as e:
            logger.error(f"Failed to create overlap text: {e}")
            # Simple fallback
            return text[-200:] if len(text) > 200 else text

    def create_all_chunks(self, processed_pages):
        """Create chunks from all processed pages"""
        logger.info(f"Starting chunk creation for {len(processed_pages)} pages")
        start_time = time.time()

        try:
            all_chunks = []
            total_tokens = 0

            for i, page_data in enumerate(processed_pages):
                page_start = time.time()
                logger.info(f"Creating chunks for page {page_data['page_number']} ({i + 1}/{len(processed_pages)})")

                page_chunks = self.create_chunks_from_page(page_data, page_data["page_number"])
                all_chunks.extend(page_chunks)

                page_tokens = sum(chunk["token_count"] for chunk in page_chunks)
                total_tokens += page_tokens

                page_time = time.time() - page_start
                logger.debug(f"Page {page_data['page_number']}: {len(page_chunks)} chunks, "
                             f"{page_tokens} tokens in {page_time:.3f}s")

            total_time = time.time() - start_time
            avg_tokens_per_chunk = total_tokens / len(all_chunks) if all_chunks else 0

            logger.info(f"Chunk creation completed:")
            logger.info(f"  - Total chunks: {len(all_chunks)}")
            logger.info(f"  - Total tokens: {total_tokens:,}")
            logger.info(f"  - Average tokens per chunk: {avg_tokens_per_chunk:.0f}")
            logger.info(f"  - Processing time: {total_time:.2f}s")
            logger.info(f"  - Average time per page: {total_time / len(processed_pages):.3f}s")

            # Validate chunk quality
            small_chunks = len([c for c in all_chunks if c["token_count"] < CHUNK_SIZE * 0.1])
            if small_chunks > 0:
                logger.warning(f"{small_chunks} chunks are very small (< 10% of target size)")

            large_chunks = len([c for c in all_chunks if c["token_count"] > CHUNK_SIZE * 1.1])
            if large_chunks > 0:
                logger.warning(f"{large_chunks} chunks exceed target size by >10%")

            return all_chunks

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to create chunks after {processing_time:.2f}s: {e}", exc_info=True)
            raise