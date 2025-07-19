import os
import logging
from logging.handlers import RotatingFileHandler

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
LOG_DIR = "./logs/"

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)


# Setup logging with both file and console output
def setup_logging():
    # Create logger
    logger = logging.getLogger("financial_agent")
    logger.setLevel(logging.INFO)  # Change this if Debugging

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # File handler for all logs (with rotation)
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "app.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Separate file handler for errors only
    error_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "errors.log"),
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)

    # Console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Less verbose for console
    console_handler.setFormatter(simple_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger


# Setup logging and create logger instance
logger = setup_logging()

# Test logging setup
logger.info("Logging system initialized")
logger.debug(f"Log directory: {LOG_DIR}")

# Optional: Set specific log levels for noisy libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)