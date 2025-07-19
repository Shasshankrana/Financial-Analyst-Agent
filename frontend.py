import streamlit as st
import requests


st.set_page_config(
    page_title="Financial Analyst AI Agent",
    page_icon="📈",
    layout="wide"
)

# Backend URL
BACKEND_URL = "http://localhost:8000"


def check_backend():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def main():
    st.title("📈 Financial Analyst AI Agent")

    # Check backend status
    if not check_backend():
        st.error("❌ Backend not running. Start with: `uvicorn backend:app --reload`")
        return

    st.success("✅ Backend connected")

    # Sidebar
    st.sidebar.header("Configuration")

    # Document upload
    st.sidebar.subheader("📄 Document Upload")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=['pdf'])

    if uploaded_file and st.sidebar.button("Process Document"):
        with st.sidebar.spinner("Processing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{BACKEND_URL}/upload", files=files)

            if response.status_code == 200:
                result = response.json()
                if result["status"] == "success":
                    st.sidebar.success(f"✅ Processed {result['chunks']} chunks")
                    st.session_state.document_loaded = True
                else:
                    st.sidebar.error(f"❌ {result['message']}")
            else:
                st.sidebar.error("❌ Upload failed")

    # Data source selection
    source = st.sidebar.selectbox(
        "Data Source",
        ["auto", "pdf_only", "api_only", "both"],
        help="Choose which data sources to use"
    )

    # Document status
    if st.session_state.get('document_loaded'):
        st.sidebar.info("📄 Document loaded")
    else:
        st.sidebar.warning("📄 No document loaded")

    # Main interface
    st.header("💬 Chat Interface")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                st.caption(f"Sources: {', '.join(message['sources'])}")

    # Chat input
    if prompt := st.chat_input("Ask about NVIDIA's financials or stock price..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/query",
                        json={"query": prompt, "source": source},
                        timeout=60
                    )

                    if response.status_code == 200:
                        result = response.json()

                        if result["status"] == "success":
                            answer = result["answer"]
                            sources = result.get("sources", [])

                            st.write(answer)
                            if sources:
                                st.caption(f"Sources: {', '.join(sources)}")

                            # Add to chat history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": sources
                            })
                        else:
                            error_msg = result.get("message", "Unknown error")
                            st.error(f"❌ {error_msg}")
                    else:
                        st.error("❌ Server error")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # Example queries
    st.header("💡 Example Queries")
    examples = [
        "What are NVIDIA's main risk factors?",
        "What is NVDA's current stock price?",
        "Summarize the Data Center business and show current stock price"
    ]

    for i, example in enumerate(examples):
        if st.button(example, key=f"example_{i}"):
            # Trigger the example query
            st.session_state.messages.append({"role": "user", "content": example})
            st.rerun()
