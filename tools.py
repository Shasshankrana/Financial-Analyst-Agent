import aiohttp
import time
import google.generativeai as genai
from vector_store import VectorStore
from document_processor import DocumentProcessor
from config import GEMINI_API_KEY, ALPHA_VANTAGE_API_KEY, RAG_PROMPT, logger

genai.configure(api_key=GEMINI_API_KEY)


class RAGTool:
    def __init__(self):
        logger.info("Initializing RAGTool...")
        start_time = time.time()

        try:
            # Initialize vector store
            logger.debug("Creating VectorStore instance...")
            vector_start = time.time()
            self.vector_store = VectorStore()
            vector_time = time.time() - vector_start
            logger.info(f"VectorStore initialized in {vector_time:.2f}s")

            # Initialize document processor
            logger.debug("Creating DocumentProcessor instance...")
            doc_start = time.time()
            self.doc_processor = DocumentProcessor()
            doc_time = time.time() - doc_start
            logger.info(f"DocumentProcessor initialized in {doc_time:.2f}s")

            # Initialize Gemini model
            logger.debug("Creating Gemini model instance...")
            model_start = time.time()
            self.model = genai.GenerativeModel("gemini-2.0-flash")
            model_time = time.time() - model_start
            logger.info(f"Gemini model initialized in {model_time:.2f}s")

            self.is_ready = False

            total_time = time.time() - start_time
            logger.info(f"RAGTool initialization completed:")
            logger.info(f"  - VectorStore: {vector_time:.2f}s")
            logger.info(f"  - DocumentProcessor: {doc_time:.2f}s")
            logger.info(f"  - Gemini model: {model_time:.2f}s")
            logger.info(f"  - Total time: {total_time:.2f}s")
            logger.info(f"  - Status: Ready for document loading")

        except Exception as e:
            init_time = time.time() - start_time
            logger.error(f"RAGTool initialization failed after {init_time:.2f}s: {e}", exc_info=True)
            raise

    async def load_document(self, pdf_path):
        """Load and process document with page-level processing"""
        logger.info(f"📄 Starting document loading: {pdf_path}")
        start_time = time.time()

        try:
            # Validate input
            logger.debug(f"Document path: {pdf_path}")
            if not pdf_path:
                logger.error("No PDF path provided")
                return {"status": "error", "message": "No PDF path provided"}

            # Phase 1: Document processing
            logger.info("📋 Phase 1: Processing document pages with vision...")
            processing_start = time.time()

            processed_pages = await self.doc_processor.process_document(pdf_path)
            processing_time = time.time() - processing_start

            if not processed_pages:
                logger.error("Document processing returned no pages")
                return {"status": "error", "message": "No pages processed from document"}

            logger.info(f"✅ Document processing completed:")
            logger.info(f"  - Pages processed: {len(processed_pages)}")
            logger.info(f"  - Processing time: {processing_time:.2f}s")
            logger.info(f"  - Average time per page: {processing_time / len(processed_pages):.2f}s")

            # Phase 2: Chunk creation
            logger.info("🔨 Phase 2: Creating chunks from processed pages...")
            chunking_start = time.time()

            chunks = self.doc_processor.create_all_chunks(processed_pages)
            chunking_time = time.time() - chunking_start

            if not chunks:
                logger.error("Chunk creation returned no chunks")
                return {"status": "error", "message": "No chunks created from document"}

            # Log chunk statistics
            chunk_tokens = [chunk.get("token_count", 0) for chunk in chunks]
            total_tokens = sum(chunk_tokens)
            avg_tokens = total_tokens / len(chunks) if chunks else 0

            logger.info(f"✅ Chunk creation completed:")
            logger.info(f"  - Total chunks: {len(chunks)}")
            logger.info(f"  - Total tokens: {total_tokens:,}")
            logger.info(f"  - Average tokens per chunk: {avg_tokens:.0f}")
            logger.info(f"  - Chunking time: {chunking_time:.2f}s")

            # Phase 3: Vector index building
            logger.info("🔍 Phase 3: Building vector search index...")
            indexing_start = time.time()

            self.vector_store.build_index(chunks)
            indexing_time = time.time() - indexing_start

            logger.info(f"✅ Vector index built in {indexing_time:.2f}s")

            # Update ready status
            self.is_ready = True
            logger.info("🎯 RAGTool is now ready for queries")

            # Get comprehensive statistics
            stats = self.vector_store.get_stats()
            total_time = time.time() - start_time

            # Log final summary
            logger.info(f"📊 Document loading completed successfully:")
            logger.info(f"  - Total time: {total_time:.2f}s")
            logger.info(f"  - Processing: {processing_time:.2f}s ({processing_time / total_time * 100:.1f}%)")
            logger.info(f"  - Chunking: {chunking_time:.2f}s ({chunking_time / total_time * 100:.1f}%)")
            logger.info(f"  - Indexing: {indexing_time:.2f}s ({indexing_time / total_time * 100:.1f}%)")
            logger.info(f"  - Pages processed: {stats.get('unique_pages', 0)}")
            logger.info(f"  - Final chunks: {stats.get('total_chunks', 0)}")
            logger.info(f"  - Searchable tokens: {stats.get('total_tokens', 0):,}")

            return {
                "status": "success",
                "message": f"Processed {stats['unique_pages']} pages into {stats['total_chunks']} chunks",
                "chunks": stats['total_chunks'],
                "stats": stats,
                "timing": {
                    "total": total_time,
                    "processing": processing_time,
                    "chunking": chunking_time,
                    "indexing": indexing_time
                }
            }

        except Exception as e:
            load_time = time.time() - start_time
            logger.error(f"❌ Document loading failed after {load_time:.2f}s", exc_info=True)
            logger.error(f"Failed document: {pdf_path}")

            return {
                "status": "error",
                "message": f"Document loading failed: {str(e)}",
                "load_time": load_time
            }

    async def query_with_streaming(self, question):
        """Query the document using enhanced RAG with streaming LLM response"""
        logger.info(f"🔍 Starting RAG query with streaming: '{question[:100]}...'")
        start_time = time.time()

        # Validate readiness
        if not self.is_ready:
            logger.warning("❌ RAG query attempted but no document loaded")
            yield {"type": "error", "message": "No document loaded"}
            return

        try:
            # Log query details
            logger.info(f"Query details:")
            logger.info(f"  - Question: '{question}'")
            logger.info(f"  - Question length: {len(question)} characters")
            logger.info(f"  - RAG status: Ready")

            # Phase 1: Vector search
            logger.info("🔎 Phase 1: Searching for relevant chunks...")
            search_start = time.time()

            similar_chunks = self.vector_store.search(question, k=10)
            search_time = time.time() - search_start

            if not similar_chunks:
                logger.warning("⚠️ No relevant chunks found for query")
                yield {"type": "error", "message": "No relevant information found in the document"}
                return

            logger.info(f"✅ Vector search completed:")
            logger.info(f"  - Chunks found: {len(similar_chunks)}")
            logger.info(f"  - Search time: {search_time:.3f}s")

            # Notify frontend that search is complete
            yield {"type": "search_complete", "chunks_found": len(similar_chunks)}

            # Log search quality
            scores = [score for _, score in similar_chunks]
            best_score = max(scores)
            worst_score = min(scores)
            avg_score = sum(scores) / len(scores)

            logger.info(f"  - Score range: {worst_score:.3f} - {best_score:.3f}")
            logger.info(f"  - Average score: {avg_score:.3f}")

            # Phase 2: Context organization
            logger.info("📋 Phase 2: Organizing context by pages...")
            org_start = time.time()

            # Group chunks by page for better context
            page_contexts = {}
            for chunk, score in similar_chunks:
                page_num = chunk["page_number"]
                if page_num not in page_contexts:
                    page_contexts[page_num] = []
                page_contexts[page_num].append((chunk, score))

            logger.debug(f"Chunks grouped into {len(page_contexts)} pages")

            # Build rich context
            context_parts = []
            sources = []
            total_context_length = 0

            for page_num, page_chunks in page_contexts.items():
                # Sort chunks by chunk index for proper order
                page_chunks.sort(key=lambda x: x[0].get("chunk_index", 0))

                page_text = ""
                best_score = max(score for _, score in page_chunks)
                chunk_count = len(page_chunks)

                for chunk, score in page_chunks:
                    page_text += chunk["text"] + " "

                page_context = f"[Page {page_num} - Relevance: {best_score:.3f} - Chunks: {chunk_count}]:\n{page_text.strip()}"
                context_parts.append(page_context)
                sources.append(f"Page {page_num}")
                total_context_length += len(page_context)

            context = "\n\n".join(context_parts)
            org_time = time.time() - org_start

            logger.info(f"✅ Context organization completed:")
            logger.info(f"  - Pages referenced: {len(page_contexts)}")
            logger.info(f"  - Total context length: {total_context_length:,} characters")
            logger.info(f"  - Organization time: {org_time:.3f}s")

            # Send sources to frontend
            unique_sources = list(set(sources))
            yield {"type": "sources", "sources": unique_sources}

            # Phase 3: Gemini response generation with streaming
            logger.info("🤖 Phase 3: Generating streaming response with Gemini...")
            generation_start = time.time()

            # Create the full prompt
            full_prompt = RAG_PROMPT.format(context=context, question=question)
            prompt_length = len(full_prompt)

            logger.debug(f"Generated prompt: {prompt_length:,} characters")

            if prompt_length > 30000:  # Gemini context length warning
                logger.warning(f"⚠️ Large prompt size: {prompt_length:,} characters - might hit context limits")

            # Notify frontend that LLM generation is starting
            yield {"type": "llm_start"}

            # Call Gemini with streaming
            gemini_start = time.time()

            try:
                response = self.model.generate_content(
                    full_prompt,
                    stream=True,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=2048,
                    )
                )

                full_answer = ""
                for chunk in response:
                    if chunk.text:
                        full_answer += chunk.text
                        # Stream each chunk to frontend
                        yield {"type": "llm_chunk", "text": chunk.text}

                gemini_time = time.time() - gemini_start
                generation_time = time.time() - generation_start
                total_time = time.time() - start_time

                # Validate response
                if not full_answer or len(full_answer.strip()) < 10:
                    logger.warning("⚠️ Gemini returned very short or empty response")

                # Log final results
                logger.info(f"✅ RAG query with streaming completed successfully:")
                logger.info(f"  - Total time: {total_time:.2f}s")
                logger.info(f"  - Search: {search_time:.3f}s ({search_time / total_time * 100:.1f}%)")
                logger.info(f"  - Organization: {org_time:.3f}s ({org_time / total_time * 100:.1f}%)")
                logger.info(f"  - Generation: {generation_time:.2f}s ({generation_time / total_time * 100:.1f}%)")
                logger.info(f"  - Answer length: {len(full_answer):,} characters")
                logger.info(f"  - Sources used: {len(unique_sources)} ({', '.join(unique_sources)})")

            except Exception as genai_error:
                logger.error(f"❌ Gemini API error: {genai_error}")
                yield {"type": "error", "message": f"AI generation failed: {str(genai_error)}"}
                return

        except Exception as e:
            query_time = time.time() - start_time
            logger.error(f"❌ RAG query failed after {query_time:.2f}s", exc_info=True)
            logger.error(f"Failed query: '{question[:100]}...'")
            yield {"type": "error", "message": f"RAG query failed: {str(e)}"}

    async def query(self, question):
        """Non-streaming version for backward compatibility"""
        logger.info(f"🔍 Starting non-streaming RAG query: '{question[:100]}...'")

        full_answer = ""
        sources = []

        async for chunk in self.query_with_streaming(question):
            if chunk.get("type") == "llm_chunk":
                full_answer += chunk.get("text", "")
            elif chunk.get("type") == "sources":
                sources = chunk.get("sources", [])
            elif chunk.get("type") == "error":
                return {"status": "error", "message": chunk.get("message", "Unknown error")}

        return {
            "status": "success",
            "answer": full_answer,
            "sources": sources
        }


class StockTool:
    def __init__(self):
        logger.info("Initializing StockTool...")
        start_time = time.time()

        try:
            self.api_key = ALPHA_VANTAGE_API_KEY
            self.base_url = "https://www.alphavantage.co/query"

            # Validate API key
            if not self.api_key:
                logger.error("❌ ALPHA_VANTAGE_API_KEY not found in configuration")
                raise ValueError("Alpha Vantage API key not configured")

            # Mask API key for logging (show only first and last few characters)
            masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}" if len(self.api_key) > 8 else "***"

            init_time = time.time() - start_time
            logger.info(f"✅ StockTool initialized successfully:")
            logger.info(f"  - API Key: {masked_key}")
            logger.info(f"  - Base URL: {self.base_url}")
            logger.info(f"  - Initialization time: {init_time:.3f}s")
            logger.info(f"  - Status: Ready for stock queries")

        except Exception as e:
            init_time = time.time() - start_time
            logger.error(f"❌ StockTool initialization failed after {init_time:.3f}s: {e}", exc_info=True)
            raise

    async def get_price(self, symbol):
        """Get current stock price"""
        logger.info(f"📈 Fetching stock data for symbol: {symbol}")
        start_time = time.time()

        try:
            # Validate input
            if not symbol or not symbol.strip():
                logger.error("❌ No stock symbol provided")
                return {"status": "error", "message": "No stock symbol provided"}

            symbol = symbol.strip().upper()
            logger.debug(f"Normalized symbol: {symbol}")

            # Prepare API request
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.api_key
            }

            logger.debug(
                f"API request parameters: {dict((k, v if k != 'apikey' else '***') for k, v in params.items())}")

            # Make API call
            logger.info(f"🌐 Calling Alpha Vantage API...")
            api_start = time.time()

            async with aiohttp.ClientSession() as session:
                logger.debug(f"Making GET request to: {self.base_url}")

                async with session.get(self.base_url, params=params) as response:
                    response_start = time.time()

                    # Log response details
                    status_code = response.status
                    content_type = response.headers.get('content-type', 'unknown')

                    logger.debug(f"API response received:")
                    logger.debug(f"  - Status code: {status_code}")
                    logger.debug(f"  - Content type: {content_type}")

                    if status_code != 200:
                        logger.error(f"❌ API returned non-200 status: {status_code}")
                        return {"status": "error", "message": f"API error: HTTP {status_code}"}

                    # Parse response
                    try:
                        data = await response.json()
                        parse_time = time.time() - response_start
                        logger.debug(f"JSON parsing completed in {parse_time:.3f}s")
                    except Exception as parse_error:
                        logger.error(f"❌ Failed to parse JSON response: {parse_error}")
                        return {"status": "error", "message": "Invalid JSON response from API"}

            api_time = time.time() - api_start
            logger.info(f"✅ API call completed in {api_time:.2f}s")

            # Process response data
            logger.debug("Processing API response data...")

            # Log response structure (without sensitive data)
            response_keys = list(data.keys()) if isinstance(data, dict) else "non-dict"
            logger.debug(f"Response keys: {response_keys}")

            # Check for API errors
            if "Error Message" in data:
                error_msg = data["Error Message"]
                logger.error(f"❌ Alpha Vantage API error: {error_msg}")
                return {"status": "error", "message": f"API error: {error_msg}"}

            if "Note" in data:
                note_msg = data["Note"]
                logger.warning(f"⚠️ Alpha Vantage API note: {note_msg}")
                if "call frequency" in note_msg.lower():
                    return {"status": "error", "message": "API rate limit exceeded. Please try again later."}

            # Extract stock data
            if "Global Quote" in data:
                quote = data["Global Quote"]
                logger.debug(f"Found Global Quote data with {len(quote)} fields")

                # Extract and validate fields
                try:
                    symbol_result = quote.get("01. symbol", symbol)
                    price_str = quote.get("05. price", "0")
                    change_percent = quote.get("10. change percent", "0%")
                    volume_str = quote.get("06. volume", "0")

                    # Convert and validate numeric fields
                    try:
                        price = float(price_str)
                        volume = int(float(volume_str))  # Convert to float first to handle scientific notation
                    except (ValueError, TypeError) as convert_error:
                        logger.error(f"❌ Failed to convert numeric fields: {convert_error}")
                        logger.debug(f"Raw data - price: '{price_str}', volume: '{volume_str}'")
                        return {"status": "error", "message": "Invalid numeric data from API"}

                    # Validate data quality
                    if price <= 0:
                        logger.warning(f"⚠️ Unusual price value: ${price}")

                    if volume < 0:
                        logger.warning(f"⚠️ Unusual volume value: {volume:,}")

                    result = {
                        "status": "success",
                        "symbol": symbol_result,
                        "price": price,
                        "change": change_percent,
                        "volume": volume
                    }

                    total_time = time.time() - start_time

                    # Log successful result
                    logger.info(f"✅ Stock data retrieved successfully:")
                    logger.info(f"  - Symbol: {symbol_result}")
                    logger.info(f"  - Price: ${price:.2f}")
                    logger.info(f"  - Change: {change_percent}")
                    logger.info(f"  - Volume: {volume:,}")
                    logger.info(f"  - API time: {api_time:.2f}s")
                    logger.info(f"  - Total time: {total_time:.2f}s")

                    # Data quality checks
                    if price > 1000:
                        logger.info(f"  - High price stock (>${price:.2f})")
                    if volume > 1000000:
                        logger.info(f"  - High volume trading ({volume:,} shares)")

                    return result

                except KeyError as key_error:
                    logger.error(f"❌ Missing expected field in API response: {key_error}")
                    logger.debug(f"Available quote fields: {list(quote.keys())}")
                    return {"status": "error", "message": f"Missing field in API response: {key_error}"}

            else:
                logger.error("❌ No 'Global Quote' found in API response")
                logger.debug(f"Response structure: {list(data.keys()) if isinstance(data, dict) else type(data)}")

                # Check for other possible response formats
                if "Time Series" in str(data):
                    logger.debug("Response appears to contain time series data instead of quote")

                return {"status": "error", "message": "Unexpected API response format - no quote data found"}

        except aiohttp.ClientError as client_error:
            api_time = time.time() - start_time
            logger.error(f"❌ HTTP client error after {api_time:.2f}s: {client_error}")
            return {"status": "error", "message": f"Network error: {str(client_error)}"}

        except Exception as e:
            query_time = time.time() - start_time
            logger.error(f"❌ Stock API query failed after {query_time:.2f}s", exc_info=True)
            logger.error(f"Failed symbol: {symbol}")

            return {
                "status": "error",
                "message": f"Stock API failed: {str(e)}",
                "query_time": query_time
            }