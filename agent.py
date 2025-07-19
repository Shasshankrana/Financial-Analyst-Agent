import asyncio
import re
from enum import Enum
from tools import RAGTool, StockTool
from config import logger
import time


class DataSource(Enum):
    AUTO = "auto"
    PDF_ONLY = "pdf_only"
    API_ONLY = "api_only"
    BOTH = "both"


class FinancialAgent:
    def __init__(self):
        logger.info("🤖 Initializing Financial Agent...")
        start_time = time.time()

        try:
            # Initialize tools
            logger.info("📚 Initializing RAG Tool...")
            rag_start = time.time()
            self.rag_tool = RAGTool()
            rag_time = time.time() - rag_start
            logger.info(f"✅ RAG Tool initialized in {rag_time:.2f}s")

            logger.info("📈 Initializing Stock Tool...")
            stock_start = time.time()
            self.stock_tool = StockTool()
            stock_time = time.time() - stock_start
            logger.info(f"✅ Stock Tool initialized in {stock_time:.2f}s")

            # Agent state
            self.document_loaded = False

            total_time = time.time() - start_time
            logger.info(f"🎯 Financial Agent initialization completed:")
            logger.info(f"  - RAG Tool: {rag_time:.2f}s")
            logger.info(f"  - Stock Tool: {stock_time:.2f}s")
            logger.info(f"  - Total time: {total_time:.2f}s")
            logger.info(f"  - Status: Ready for operations")

        except Exception as e:
            init_time = time.time() - start_time
            logger.error(f"❌ Financial Agent initialization failed after {init_time:.2f}s: {e}", exc_info=True)
            raise

    async def load_document(self, pdf_path):
        """Load a PDF document for RAG queries"""
        logger.info(f"📄 Agent loading document: {pdf_path}")
        start_time = time.time()

        try:
            result = await self.rag_tool.load_document(pdf_path)

            if result.get("status") == "success":
                self.document_loaded = True
                load_time = time.time() - start_time
                logger.info(f"✅ Document loaded successfully by agent in {load_time:.2f}s")
                logger.info(f"  - Document status: Ready for queries")
                logger.info(f"  - Chunks processed: {result.get('chunks', 'unknown')}")
            else:
                load_time = time.time() - start_time
                logger.error(f"❌ Document loading failed after {load_time:.2f}s")
                logger.error(f"  - Error: {result.get('message', 'Unknown error')}")
                self.document_loaded = False

            return result

        except Exception as e:
            load_time = time.time() - start_time
            logger.error(f"❌ Agent document loading failed after {load_time:.2f}s: {e}", exc_info=True)
            self.document_loaded = False
            return {
                "status": "error",
                "message": f"Agent document loading failed: {str(e)}",
                "load_time": load_time
            }

    def _extract_stock_symbols(self, query):
        """Extract stock symbols from query"""
        # Common patterns for stock symbols
        patterns = [
            r'\b([A-Z]{1,5})\b(?:\s+stock|\s+price|\s+share)',  # Symbol followed by stock/price/share
            r'\$([A-Z]{1,5})\b',  # $SYMBOL format
            r'\b(NVDA|NVIDIA)\b',  # Specific common symbols
            r'\b([A-Z]{2,5})\s+current\s+price',  # Symbol current price
        ]

        symbols = set()
        query_upper = query.upper()

        for pattern in patterns:
            matches = re.findall(pattern, query_upper)
            for match in matches:
                if isinstance(match, tuple):
                    symbols.update(match)
                else:
                    symbols.add(match)

        # Clean up symbols
        cleaned_symbols = set()
        for symbol in symbols:
            if symbol and len(symbol) >= 2 and symbol.isalpha():
                cleaned_symbols.add(symbol)

        logger.debug(f"Extracted stock symbols from query: {list(cleaned_symbols)}")
        return list(cleaned_symbols)

    def _determine_data_sources(self, query, data_source):
        """Determine which data sources to use based on query and settings"""
        logger.debug(f"Determining data sources for query: '{query[:50]}...'")
        logger.debug(f"Data source setting: {data_source}")

        # Extract stock symbols from query
        stock_symbols = self._extract_stock_symbols(query)
        has_stock_query = bool(stock_symbols) or any(
            keyword in query.lower()
            for keyword in ['stock price', 'current price', 'share price', 'market cap', 'trading']
        )

        # Determine if document query
        has_doc_query = any(
            keyword in query.lower()
            for keyword in ['risk', 'revenue', 'business', 'financial', 'earnings', 'segment', 'summarize']
        )

        use_rag = False
        use_stock = False

        if data_source == DataSource.AUTO:
            use_rag = has_doc_query and self.document_loaded
            use_stock = has_stock_query
        elif data_source == DataSource.PDF_ONLY:
            use_rag = self.document_loaded
            use_stock = False
        elif data_source == DataSource.API_ONLY:
            use_rag = False
            use_stock = True
        elif data_source == DataSource.BOTH:
            use_rag = self.document_loaded
            use_stock = True

        logger.info(f"Data source decision:")
        logger.info(f"  - Use RAG: {use_rag} (doc_loaded: {self.document_loaded}, has_doc_query: {has_doc_query})")
        logger.info(f"  - Use Stock API: {use_stock} (has_stock_query: {has_stock_query})")
        logger.info(f"  - Stock symbols found: {stock_symbols}")

        return use_rag, use_stock, stock_symbols

    async def query_with_streaming(self, query, data_source):
        """Process query with streaming LLM response"""
        logger.info(f"🔍 Agent processing streaming query: '{query[:100]}...'")
        start_time = time.time()

        try:
            # Convert string to enum if needed
            if isinstance(data_source, str):
                data_source = DataSource(data_source)

            # Determine what data sources to use
            use_rag, use_stock, stock_symbols = self._determine_data_sources(query, data_source)

            if not use_rag and not use_stock:
                logger.warning("⚠️ No data sources available for query")
                yield {"type": "error",
                       "message": "No data sources available. Please load a document or enable stock API."}
                return

            # Collect data from sources
            rag_data = None
            stock_data = {}

            # Get stock data if needed (this is fast, so we do it first)
            if use_stock and stock_symbols:
                logger.info(f"📈 Fetching stock data for symbols: {stock_symbols}")
                for symbol in stock_symbols:
                    try:
                        result = await self.stock_tool.get_price(symbol)
                        if result.get("status") == "success":
                            stock_data[symbol] = result
                            logger.info(f"✅ Got stock data for {symbol}: ${result.get('price', 'N/A')}")
                        else:
                            logger.warning(
                                f"⚠️ Failed to get stock data for {symbol}: {result.get('message', 'Unknown error')}")
                    except Exception as e:
                        logger.error(f"❌ Stock API error for {symbol}: {e}")

            # Process with RAG if needed (this includes streaming)
            if use_rag:
                logger.info("📚 Processing with RAG (streaming)...")

                # If we have stock data, enhance the query
                enhanced_query = query
                if stock_data:
                    stock_info = []
                    for symbol, data in stock_data.items():
                        price = data.get('price', 'N/A')
                        change = data.get('change', 'N/A')
                        stock_info.append(f"{symbol}: ${price} ({change})")

                    stock_context = "Current stock prices: " + ", ".join(stock_info)
                    enhanced_query = f"{query}\n\nCurrent market data: {stock_context}"
                    logger.debug(f"Enhanced query with stock data: {len(enhanced_query)} characters")

                # Stream the RAG response
                async for chunk in self.rag_tool.query_with_streaming(enhanced_query):
                    yield chunk

            elif use_stock:
                # Only stock data, create a simple response
                logger.info("📈 Creating stock-only response...")
                yield {"type": "llm_start"}

                if stock_data:
                    response_parts = []
                    for symbol, data in stock_data.items():
                        price = data.get('price', 'N/A')
                        change = data.get('change', 'N/A')
                        volume = data.get('volume', 'N/A')
                        response_parts.append(
                            f"{symbol} is currently trading at ${price} with a change of {change}. Volume: {volume:,} shares.")

                    full_response = "Here's the current stock information:\n\n" + "\n".join(response_parts)

                    # Stream word by word for consistency
                    words = full_response.split()
                    for word in words:
                        yield {"type": "llm_chunk", "text": word + " "}
                        await asyncio.sleep(0.05)  # Slightly slower for readability
                else:
                    error_response = "I couldn't retrieve stock data for the requested symbols. Please check the symbol and try again."
                    words = error_response.split()
                    for word in words:
                        yield {"type": "llm_chunk", "text": word + " "}
                        await asyncio.sleep(0.05)

            total_time = time.time() - start_time
            logger.info(f"✅ Agent streaming query completed in {total_time:.2f}s")

        except Exception as e:
            query_time = time.time() - start_time
            logger.error(f"❌ Agent streaming query failed after {query_time:.2f}s: {e}", exc_info=True)
            yield {"type": "error", "message": f"Agent query failed: {str(e)}"}

    async def query(self, query, data_source):
        """Non-streaming version for backward compatibility"""
        logger.info(f"🔍 Agent processing non-streaming query: '{query[:100]}...'")

        full_answer = ""
        sources = []

        async for chunk in self.query_with_streaming(query, data_source):
            if chunk.get("type") == "llm_chunk":
                full_answer += chunk.get("text", "")
            elif chunk.get("type") == "sources":
                sources = chunk.get("sources", [])
            elif chunk.get("type") == "error":
                return {"status": "error", "message": chunk.get("message", "Unknown error")}

        return {
            "status": "success",
            "answer": full_answer.strip(),
            "sources": sources
        }

    def get_status(self):
        """Get agent status"""
        return {
            "document_loaded": self.document_loaded,
            "rag_ready": self.rag_tool.is_ready if hasattr(self.rag_tool, 'is_ready') else False,
            "stock_api_ready": True  # Stock API is always ready if initialized
        }