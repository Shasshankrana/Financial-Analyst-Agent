import os
import logging
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Embedding Model Settings
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions, 512 max tokens
EMBEDDING_DIMENSION = 384
MAX_TOKENS = 512

# Chunking Settings (optimized for bge-small-en-v1.5)
CHUNK_SIZE = 400  # tokens (leaving buffer for tokenization)
CHUNK_OVERLAP = 80  # tokens (20% overlap)

RAG_PROMPT = """
Based on the following context from a financial document, provide a comprehensive and accurate answer to the question.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Provide a detailed, well-structured answer based on the context
2. Include specific financial figures, dates, and facts when available
3. Reference page numbers when citing specific information
4. If the context doesn't contain complete information, clearly state what's missing
5. Organize your response logically with clear sections if dealing with complex topics
6. Preserve the precision of any financial data or technical details

ANSWER:
"""

OCR_PROMPT = """
Analyze this page from a financial document and extract ALL information in detailed natural language.

INSTRUCTIONS:
1. **Text Content**: Transcribe all readable text, preserving structure and meaning
2. **Tables & Data**: Convert all tables into detailed natural language descriptions, including:
   - What the table represents
   - All column headers and row data
   - Relationships between data points
   - Financial figures with their context
3. **Charts & Graphs**: Describe any visual data representations:
   - Type of chart (bar, line, pie, etc.)
   - What metrics are being shown
   - Key trends and data points
   - Axis labels and values
4. **Visual Elements**: Describe any important visual elements, logos, or formatting that adds context
5. **Financial Context**: For any numbers or percentages, explain what they represent

OUTPUT FORMAT:
- Write in clear, descriptive sentences
- Preserve the logical flow and structure of information
- Make implicit relationships explicit
- Use natural language that would be useful for semantic search

Focus on creating rich, searchable content that captures the complete meaning of this page.
"""

# Paths
VECTOR_STORE_PATH = "./vector_store/"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)