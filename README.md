# Financial Analyst AI Agent

A streamlined AI agent that combines comprehensive document analysis with real-time stock data using LangGraph, Gemini Vision, and local embeddings.

## 🚀 Enhanced Features

- **📄 Advanced PDF Analysis**: Each page processed comprehensively with Gemini Vision
- **🧠 Local Embeddings**: Uses `BAAI/bge-small-en-v1.5` for fast, accurate semantic search
- **📊 Complete Content Extraction**: Text, tables, and charts converted to natural language
- **🔍 Page-Level Documents**: Each page treated as separate document in FAISS
- **📈 Real-time Stock Data**: Alpha Vantage integration
- **🤖 Smart Routing**: Auto-detects data source requirements

## 📋 Model Specifications

### Embedding Model: `BAAI/bge-small-en-v1.5`
- **Dimensions**: 384
- **Max Tokens**: 512
- **Optimal Chunk Size**: 400 tokens (leaving buffer)
- **Optimal Overlap**: 80 tokens (20% overlap)
- **Performance**: Excellent for financial/semantic search
- **Speed**: Fast inference on CPU

## 🛠️ Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
```

### 3. Get API Keys
- **Gemini**: [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Alpha Vantage**: [Alpha Vantage](https://www.alphavantage.co/support/#api-key)

## 🚀 Running the Application

### Start Backend
```bash
uvicorn backend:app --reload --port 8000
```

### Start Frontend  
```bash
streamlit run frontend.py --server.port 8501
```

### Access
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000

## 💡 Enhanced Document Processing

### How It Works:
1. **Page Extraction**: Each PDF page converted to high-res image (3x resolution)
2. **Vision Analysis**: Gemini Vision extracts ALL content in natural language:
   - Text transcription with context
   - Table data converted to descriptive sentences  
   - Chart/graph descriptions with data points
   - Visual elements and their meaning
3. **Smart Chunking**: Content chunked based on actual token counts
4. **Local Embeddings**: Fast, accurate embeddings using BGE model
5. **FAISS Indexing**: Each page/chunk indexed separately for precise retrieval

### Example Vision Processing:
**Input**: PDF page with revenue table

**Output**: "This page contains NVIDIA's quarterly revenue breakdown table showing Data Center revenue of 
$18.4 billion for Q2 2024, representing 87% growth year-over-year. Gaming revenue was $2.9 billion, down 33% from the previous year. The table indicates Data Center has become the dominant revenue source..."

## 🎯 Usage Examples

### Document Analysis
```
"What are NVIDIA's main risk factors?"
→ Searches across all pages, finds risk sections, provides comprehensive analysis
```

### Financial Data Extraction  
```
"What was NVIDIA's Data Center revenue growth?"
→ Finds revenue tables, extracts specific figures with context
```

### Hybrid Analysis
```
"How do NVIDIA's AI investments relate to current stock performance?"
→ Combines document insights with real-time stock data
```

## 🔧 Configuration Options

### Chunking Parameters (config.py)
```python
CHUNK_SIZE = 400        # tokens (optimal for bge-small-en-v1.5)
CHUNK_OVERLAP = 80      # tokens (20% overlap)
MAX_TOKENS = 512        # model limit
```

### Embedding Model
```python
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384
```

## 📊 Performance Benefits

### Local Embeddings vs API:
- ✅ **Faster**: No API calls for embeddings
- ✅ **Cost-effective**: No per-token charges
- ✅ **Privacy**: Document content stays local
- ✅ **Reliable**: No rate limits or network issues

### Page-Level Processing:
- ✅ **Better Context**: Preserves page-level coherence
- ✅ **Precise Sources**: Exact page references
- ✅ **Rich Content**: Tables and charts fully captured
- ✅ **Semantic Search**: Natural language descriptions improve matching

## 🧪 Testing

### Test Document Processing
```python
from document_processor import DocumentProcessor
processor = DocumentProcessor()

# Test tokenization
text = "Sample financial text"
tokens = processor.count_tokens(text)
print(f"Tokens: {tokens}")

# Test processing
pages = await processor.process_document("report.pdf")
chunks = processor.create_all_chunks(pages)
```

### Test Embeddings
```python
from vector_store import VectorStore
store = VectorStore()

# Test embedding generation
texts = ["Revenue increased 22%", "Data Center growth"]
embeddings = store.generate_embeddings(texts)
print(f"Embedding shape: {embeddings.shape}")
```

## 🔍 Troubleshooting

**Slow processing**: Reduce image resolution in DocumentProcessor
**Memory issues**: Process in smaller batches or reduce chunk size  
**Poor search results**: Increase number of chunks retrieved (k parameter)
**Token count errors**: Verify tokenizer matches embedding model

## 📈 Optimization Tips

1. **Batch Size**: Adjust embedding batch size based on available RAM
2. **Image Quality**: Balance OCR accuracy vs processing speed
3. **Chunk Overlap**: Increase overlap for better context continuity
4. **Search Results**: Tune number of retrieved chunks based on document complexity

# ================== PROJECT STRUCTURE ==================

```
financial_agent/
├── .env                     # Environment variables
├── .gitignore              # Git ignore file
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── config.py              # Configuration
├── document_processor.py  # PDF processing + OCR
├── vector_store.py        # FAISS vector store
├── tools.py               # RAG and Stock API tools
├── agent.py               # LangGraph agent
├── backend.py             # FastAPI server
└── frontend.py            # Streamlit app
```