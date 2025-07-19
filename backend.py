from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from agent import FinancialAgent, DataSource
from config import logger

app = FastAPI(title="Financial Analyst Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = FinancialAgent()


@app.get("/")
async def root():
    return {"message": "Financial Analyst Agent API"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process PDF"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files supported")

    try:
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Process document
        result = await agent.load_document(tmp_path)

        # Cleanup
        os.unlink(tmp_path)

        return result

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/query")
async def query_agent(data: dict):
    """Query the agent"""
    try:
        query = data.get("query", "")
        source = data.get("source", "auto")

        if not query:
            raise HTTPException(400, "Query required")

        data_source = DataSource(source)
        result = await agent.query(query, data_source)

        return result

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(500, str(e))