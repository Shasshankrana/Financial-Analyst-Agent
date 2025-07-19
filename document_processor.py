import fitz
import io
from PIL import Image
import google.generativeai as genai
from transformers import AutoTokenizer
from config import GEMINI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, OCR_PROMPT, logger
import re

genai.configure(api_key=GEMINI_API_KEY)


class DocumentProcessor:
    def __init__(self):
        self.vision_model = genai.GenerativeModel("gemini-1.5-pro")
        # Load tokenizer for accurate token counting
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        logger.info(f"Initialized with embedding model: {EMBEDDING_MODEL}")

    def extract_pages_as_images(self, pdf_path):
        """Extract each page as high-quality image"""
        doc = fitz.open(pdf_path)
        pages = []

        logger.info(f"Extracting {doc.page_count} pages as images")

        for page_num in range(doc.page_count):
            page = doc[page_num]

            # Convert to high-resolution image (3x for better OCR)
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            img_data = pix.tobytes("png")

            pages.append({
                "page_number": page_num + 1,
                "image_data": img_data,
                "width": pix.width,
                "height": pix.height
            })

        doc.close()
        return pages

    async def extract_comprehensive_content(self, image_data, page_number):
        """Use Gemini Vision to extract ALL content in natural language"""
        try:
            image = Image.open(io.BytesIO(image_data))

            response = await self.vision_model.generate_content_async([OCR_PROMPT, image])
            extracted_text = response.text

            logger.info(f"Extracted {len(extracted_text)} characters from page {page_number}")
            return extracted_text

        except Exception as e:
            logger.error(f"Vision processing failed for page {page_number}: {e}")
            return f"Error processing page {page_number}: Content could not be extracted"

    async def process_document(self, pdf_path):
        """Process entire document - each page becomes one document"""
        logger.info(f"Processing PDF: {pdf_path}")

        # Extract all pages as images
        pages = self.extract_pages_as_images(pdf_path)

        # Process each page with vision model
        processed_pages = []
        for page_data in pages:
            page_num = page_data["page_number"]
            logger.info(f"Processing page {page_num} with vision model...")

            # Extract comprehensive content using vision
            content = await self.extract_comprehensive_content(
                page_data["image_data"], page_num
            )

            processed_pages.append({
                "page_number": page_num,
                "content": content,
                "image_dimensions": {
                    "width": page_data["width"],
                    "height": page_data["height"]
                }
            })

        logger.info(f"Successfully processed {len(processed_pages)} pages")
        return processed_pages

    def count_tokens(self, text):
        """Count tokens using the actual tokenizer"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def create_chunks_from_page(self, page_content, page_number):
        """Create overlapping chunks from a single page's content"""
        content = page_content["content"]

        # Count total tokens
        total_tokens = self.count_tokens(content)

        # If content is short enough, return as single chunk
        if total_tokens <= CHUNK_SIZE:
            return [{
                "text": content,
                "page_number": page_number,
                "chunk_index": 0,
                "chunk_id": f"page_{page_number}_chunk_0",
                "token_count": total_tokens,
                "is_complete_page": True
            }]

        # Split into sentences for better chunking
        sentences = self._split_into_sentences(content)

        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If adding this sentence would exceed chunk size, save current chunk
            if current_tokens + sentence_tokens > CHUNK_SIZE and current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "chunk_id": f"page_{page_number}_chunk_{chunk_index}",
                    "token_count": current_tokens,
                    "is_complete_page": False
                })

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, CHUNK_OVERLAP)
                current_chunk = overlap_text + " " + sentence
                current_tokens = self.count_tokens(current_chunk)
                chunk_index += 1
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens

        # Add final chunk if it has content
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "page_number": page_number,
                "chunk_index": chunk_index,
                "chunk_id": f"page_{page_number}_chunk_{chunk_index}",
                "token_count": current_tokens,
                "is_complete_page": False
            })

        logger.info(f"Created {len(chunks)} chunks from page {page_number}")
        return chunks

    def _split_into_sentences(self, text):
        """Split text into sentences for better chunking"""
        # Simple sentence splitting - can be improved with spaCy if needed
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_text(self, text, overlap_tokens):
        """Get the last N tokens worth of text for overlap"""
        words = text.split()

        # Approximate: assume ~1.3 words per token (varies by tokenizer)
        overlap_words = int(overlap_tokens * 1.3)

        if len(words) <= overlap_words:
            return text

        return " ".join(words[-overlap_words:])

    def create_all_chunks(self, processed_pages):
        """Create chunks from all processed pages"""
        all_chunks = []

        for page_data in processed_pages:
            page_chunks = self.create_chunks_from_page(page_data, page_data["page_number"])
            all_chunks.extend(page_chunks)

        logger.info(f"Created total of {len(all_chunks)} chunks from {len(processed_pages)} pages")
        return all_chunks
