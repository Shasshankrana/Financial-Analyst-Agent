import aiohttp
import google.generativeai as genai
from vector_store import VectorStore
from document_processor import DocumentProcessor
from config import GEMINI_API_KEY, ALPHA_VANTAGE_API_KEY, RAG_PROMPT, logger

genai.configure(api_key=GEMINI_API_KEY)


class RAGTool:
    def __init__(self):
        self.vector_store = VectorStore()
        self.doc_processor = DocumentProcessor()
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        self.is_ready = False

    async def load_document(self, pdf_path):
        """Load and process document with page-level processing"""
        try:
            logger.info("Starting comprehensive document processing...")

            # Process each page comprehensively with vision
            processed_pages = await self.doc_processor.process_document(pdf_path)

            # Create chunks from processed pages
            chunks = self.doc_processor.create_all_chunks(processed_pages)

            # Build vector index
            self.vector_store.build_index(chunks)

            self.is_ready = True
            stats = self.vector_store.get_stats()

            logger.info("Document processing completed successfully")
            return {
                "status": "success",
                "message": f"Processed {stats['unique_pages']} pages into {stats['total_chunks']} chunks",
                "stats": stats
            }

        except Exception as e:
            logger.error(f"Document loading failed: {e}")
            return {"status": "error", "message": str(e)}

    async def query(self, question):
        """Query the document using enhanced RAG"""
        if not self.is_ready:
            return {"status": "error", "message": "No document loaded"}

        try:
            # Search for relevant chunks
            similar_chunks = self.vector_store.search(question, k=10)  # More chunks for better context

            if not similar_chunks:
                return {"status": "error", "message": "No relevant information found"}

            # Group chunks by page for better context
            page_contexts = {}
            for chunk, score in similar_chunks:
                page_num = chunk["page_number"]
                if page_num not in page_contexts:
                    page_contexts[page_num] = []
                page_contexts[page_num].append((chunk, score))

            # Build rich context
            context_parts = []
            sources = []

            for page_num, page_chunks in page_contexts.items():
                # Sort chunks by chunk index for proper order
                page_chunks.sort(key=lambda x: x[0].get("chunk_index", 0))

                page_text = ""
                best_score = max(score for _, score in page_chunks)

                for chunk, score in page_chunks:
                    page_text += chunk["text"] + " "

                context_parts.append(f"[Page {page_num} - Relevance: {best_score:.3f}]:\n{page_text.strip()}")
                sources.append(f"Page {page_num}")

            # Generate comprehensive response
            context = "\n\n".join(context_parts)


            response = await self.model.generate_content_async(RAG_PROMPT.format(context=context, question=question))

            return {
                "status": "success",
                "answer": response.text,
                "sources": list(set(sources)),  # Remove duplicates
                "chunks_used": len(similar_chunks),
                "pages_referenced": len(page_contexts)
            }

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {"status": "error", "message": str(e)}


class StockTool:
    def __init__(self):
        self.api_key = ALPHA_VANTAGE_API_KEY
        self.base_url = "https://www.alphavantage.co/query"

    async def get_price(self, symbol):
        """Get current stock price"""
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.api_key
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()

            if "Global Quote" in data:
                quote = data["Global Quote"]
                return {
                    "status": "success",
                    "symbol": quote.get("01. symbol", symbol),
                    "price": float(quote.get("05. price", 0)),
                    "change": quote.get("10. change percent", "0%"),
                    "volume": int(quote.get("06. volume", 0))
                }
            else:
                return {"status": "error", "message": "Stock data not available"}

        except Exception as e:
            logger.error(f"Stock API failed: {e}")
            return {"status": "error", "message": str(e)}