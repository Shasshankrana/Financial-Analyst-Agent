import streamlit as st
import requests
import json
import time
from config import logger

# Page config
st.set_page_config(
    page_title="Financial Analyst AI Agent",
    page_icon="📈",
    layout="wide"
)

BACKEND_URL = "http://localhost:8000"


def check_backend():
    """Check if backend is running"""
    logger.debug("Checking backend connectivity...")
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if response.status_code == 200:
            logger.info("Backend connection successful")
            return True
        else:
            logger.warning(f"Backend returned status: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Backend connection failed: {e}")
        return False


def upload_document(file):
    """Upload and process document"""
    logger.info(f"Document upload initiated: {file.name} ({file.size} bytes)")
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        logger.debug("Sending upload request to backend")
        # Increased timeout to 10 minutes for embedding generation
        response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=600)

        if response.status_code == 200:
            result = response.json()
            if result["status"] == "success":
                chunks = result.get('chunks', 'unknown')
                logger.info(f"Document processed successfully: {chunks} chunks")
                return True, f"Processed {chunks} chunks"
            else:
                error_msg = result.get('message', 'Unknown error')
                logger.error(f"Document processing failed: {error_msg}")
                return False, error_msg
        else:
            logger.error(f"Upload failed with status {response.status_code}")
            return False, f"Upload failed (Status: {response.status_code})"
    except requests.exceptions.Timeout:
        logger.error("Upload request timed out after 10 minutes")
        return False, "Processing timed out (>10 minutes). Try a smaller document or check backend logs."
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return False, f"Error: {str(e)}"


def query_agent_with_llm_streaming(query, source, message_placeholder):
    """Send query to agent and stream LLM response"""
    logger.info(f"Query with LLM streaming: '{query}' with source: '{source}'")

    try:
        # Make streaming request
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={"query": query, "source": source},
            stream=True,
            timeout=120
        )

        if response.status_code != 200:
            logger.error(f"Query failed with status {response.status_code}")
            message_placeholder.error(f"❌ Server error (Status: {response.status_code})")
            return f"Server error (Status: {response.status_code})", []

        # Process streaming response
        full_answer = ""
        sources = []
        status_shown = False
        is_streaming_text = False

        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])  # Remove "data: " prefix
                    msg_type = data.get("type", "")

                    if msg_type == "status":
                        if not status_shown:
                            message_placeholder.info(f"🔄 {data.get('message', 'Processing...')}")
                            status_shown = True

                    elif msg_type == "search_complete":
                        # Vector search completed
                        message_placeholder.info("🔍 Found relevant information, generating response...")

                    elif msg_type == "llm_start":
                        # LLM generation starting
                        message_placeholder.info("🤖 AI is thinking...")
                        is_streaming_text = True
                        full_answer = ""

                    elif msg_type == "llm_chunk":
                        # Streaming LLM text
                        if is_streaming_text:
                            chunk_text = data.get("text", "")
                            full_answer += chunk_text
                            # Update display with streaming text
                            message_placeholder.write(full_answer)

                    elif msg_type == "sources":
                        sources = data.get("sources", [])
                        logger.debug(f"Received sources: {sources}")

                    elif msg_type == "complete":
                        logger.info("Query completed successfully")
                        # Final update with sources if available
                        if sources:
                            message_placeholder.caption(f"Sources: {', '.join(sources)}")
                        break

                    elif msg_type == "error":
                        error_msg = data.get("message", "Unknown error")
                        logger.error(f"Query error: {error_msg}")
                        message_placeholder.error(f"❌ {error_msg}")
                        return error_msg, []

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse streaming data: {line[:100]}... Error: {e}")
                    continue

        # Ensure final state is displayed
        if full_answer:
            message_placeholder.write(full_answer)
            if sources:
                message_placeholder.caption(f"Sources: {', '.join(sources)}")

        logger.info(f"Query successful. Answer length: {len(full_answer)}, Sources: {len(sources)}")
        return full_answer or "No response received", sources

    except requests.exceptions.Timeout:
        logger.error("Query timed out")
        message_placeholder.error("❌ Query timed out")
        return "Query timed out", []
    except Exception as e:
        logger.error(f"Query error: {e}")
        message_placeholder.error(f"❌ Error: {str(e)}")
        return f"Error: {str(e)}", []


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    logger.debug("Initialized empty chat messages")
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False
    logger.debug("Initialized document_loaded as False")

# Main title
logger.info("Starting Streamlit frontend application")
st.title("📈 Financial Analyst AI Agent")

# Check backend
if not check_backend():
    logger.error("Backend not accessible - stopping frontend execution")
    st.error("❌ Backend not running. Start with: `uvicorn backend:app --reload`")
    st.stop()

st.success("✅ Backend connected")

# Sidebar
with st.sidebar:
    st.header("Configuration")

    # Document upload
    st.subheader("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])

    if uploaded_file:
        st.info("⚠️ Processing can take 2-5 minutes for large documents while generating embeddings.")

        with st.expander("ℹ️ What happens during processing?"):
            st.write("""
            1. **Text Extraction**: PDF content is extracted and cleaned
            2. **Text Chunking**: Document is split into manageable chunks
            3. **Embedding Generation**: AI creates vector embeddings for each chunk (this is the slow part)
            4. **Vector Storage**: Embeddings are stored for fast retrieval during queries
            """)

        if st.button("Process Document"):
            processing_status = st.empty()

            with st.spinner("Processing document... This may take several minutes for embedding generation."):
                processing_status.info("🔄 Starting document processing...")
                success, message = upload_document(uploaded_file)

                if success:
                    processing_status.empty()
                    st.success(f"✅ {message}")
                    st.session_state.document_loaded = True
                else:
                    processing_status.empty()
                    st.error(f"❌ {message}")

    # Data source selection
    source = st.selectbox(
        "Data Source",
        ["auto", "pdf_only", "api_only", "both"],
        help="Choose which data sources to use"
    )
    logger.debug(f"Data source selected: {source}")

    # Document status
    if st.session_state.document_loaded:
        st.info("📄 Document loaded")
    else:
        st.warning("📄 No document loaded")

    # Info about streaming
    st.info("🚀 LLM responses stream automatically")

# Chat interface
st.header("💬 Chat Interface")

# Display chat history
if st.session_state.messages:
    logger.debug(f"Displaying {len(st.session_state.messages)} chat messages")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            st.caption(f"Sources: {', '.join(message['sources'])}")

# Chat input
if prompt := st.chat_input("Ask about NVIDIA's financials or stock price..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Immediately rerun to show the user message
    st.rerun()

# Process the last message if it's from user and doesn't have a response yet
if (st.session_state.messages and
        st.session_state.messages[-1]["role"] == "user" and
        (len(st.session_state.messages) == 1 or st.session_state.messages[-2]["role"] == "user")):
    last_user_message = st.session_state.messages[-1]["content"]

    # Get agent response with LLM streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        answer, sources = query_agent_with_llm_streaming(last_user_message, source, message_placeholder)

        # Add to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

        # Rerun to update the chat interface
        st.rerun()

# Example queries
st.header("💡 Example Queries")

examples = [
    "What are NVIDIA's main risk factors?",
    "What is NVDA's current stock price?",
    "Summarize the Data Center business and show current stock price"
]

cols = st.columns(len(examples))
for i, example in enumerate(examples):
    with cols[i]:
        if st.button(example, key=f"example_{i}"):
            logger.info(f"Example query clicked: '{example}'")

            # Add user message
            st.session_state.messages.append({"role": "user", "content": example})

            # Process the query
            with st.spinner(f"Processing: {example}"):
                # Create a temporary placeholder for the streaming response
                temp_placeholder = st.empty()
                answer, sources = query_agent_with_llm_streaming(example, source, temp_placeholder)

            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

            # Rerun to update the chat interface
            st.rerun()

# Debug info (optional)
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.subheader("🔧 Debug Info")
    st.sidebar.json({
        "backend_url": BACKEND_URL,
        "messages_count": len(st.session_state.messages),
        "document_loaded": st.session_state.document_loaded,
        "selected_source": source
    })