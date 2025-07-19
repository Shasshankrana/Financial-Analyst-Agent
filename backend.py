from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import tempfile
import os
import json
import asyncio
from agent import FinancialAgent, DataSource
from config import logger

# Global variable to store agent
agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global agent
    logger.info("FastAPI application starting up...")
    logger.info("Initializing Financial Agent...")
    try:
        agent = FinancialAgent()
        logger.info("Financial Agent initialized successfully")
        logger.info(f"Available endpoints: {[route.path for route in app.routes]}")
    except Exception as e:
        logger.error(f"Failed to initialize Financial Agent: {e}")
        raise

    yield

    # Shutdown
    logger.info("FastAPI application shutting down...")


app = FastAPI(title="Financial Analyst Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    logger.debug("Root endpoint accessed")
    return {"message": "Financial Analyst Agent API"}


@app.get("/health")
async def health():
    logger.debug("Health check endpoint accessed")
    return {"status": "healthy"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process PDF"""
    global agent
    if agent is None:
        logger.error("Agent not initialized")
        raise HTTPException(500, "Agent not initialized")

    logger.info(f"Upload request received for file: {file.filename}")

    if not file.filename.endswith('.pdf'):
        logger.warning(f"Invalid file type uploaded: {file.filename}")
        raise HTTPException(400, "Only PDF files supported")

    try:
        logger.debug(f"Processing file: {file.filename}, size: {file.size if hasattr(file, 'size') else 'unknown'}")

        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.debug(f"Temporary file created: {tmp_path}")

        # Process document
        logger.info("Starting document processing...")
        result = await agent.load_document(tmp_path)
        logger.info(f"Document processed successfully: {result}")

        # Cleanup
        os.unlink(tmp_path)
        logger.debug(f"Temporary file cleaned up: {tmp_path}")

        return result

    except Exception as e:
        logger.error(f"Upload failed for {file.filename}: {e}", exc_info=True)
        # Cleanup on error
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.debug("Cleaned up temporary file after error")
        raise HTTPException(500, str(e))


async def generate_llm_streaming_response(query: str, data_source: DataSource):
    """Generate streaming response for LLM text generation only"""
    try:
        logger.info(f"Starting LLM streaming for query: '{query}'")

        # Send initial status
        yield f"data: {json.dumps({'type': 'status', 'message': 'Processing your query...'})}\n\n"

        # Stream the agent query with LLM streaming
        async for chunk in agent.query_with_streaming(query, data_source):
            if chunk:
                yield f"data: {json.dumps(chunk)}\n\n"

        # Send completion signal
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        logger.info("LLM streaming completed successfully")

    except Exception as e:
        logger.error(f"LLM streaming error: {e}", exc_info=True)
        error_msg = f"Error: {str(e)}"
        yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"


@app.post("/query")
async def query_agent(data: dict):
    """Query the agent with LLM streaming response"""
    global agent
    if agent is None:
        logger.error("Agent not initialized")
        raise HTTPException(500, "Agent not initialized")

    logger.info(f"Query request received: {data}")

    try:
        query = data.get("query", "")
        source = data.get("source", "auto")

        if not query:
            logger.warning("Empty query received")
            raise HTTPException(400, "Query required")

        logger.debug(f"Processing query: '{query}' with source: '{source}'")

        data_source = DataSource(source)
        logger.debug(f"DataSource created: {data_source}")

        # Return streaming response for LLM generation
        return StreamingResponse(
            generate_llm_streaming_response(query, data_source),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )

    except Exception as e:
        logger.error(f"Query failed for '{data.get('query', 'unknown')}': {e}", exc_info=True)
        raise HTTPException(500, str(e))