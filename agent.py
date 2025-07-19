from enum import Enum
from typing import Dict
from langgraph.graph import StateGraph, END
import google.generativeai as genai
from tools import RAGTool, StockTool
from config import GEMINI_API_KEY, logger

genai.configure(api_key=GEMINI_API_KEY)


class DataSource(Enum):
    AUTO = "auto"
    PDF_ONLY = "pdf_only"
    API_ONLY = "api_only"
    BOTH = "both"


class AgentState(Dict):
    query: str
    data_source: DataSource
    rag_result: Dict
    api_result: Dict
    final_answer: str
    requires_rag: bool
    requires_api: bool
    error: str


class FinancialAgent:
    def __init__(self):
        self.rag_tool = RAGTool()
        self.stock_tool = StockTool()
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()

    def _build_workflow(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(AgentState)

        workflow.add_node("analyze", self._analyze_query)
        workflow.add_node("fetch_rag", self._fetch_rag)
        workflow.add_node("fetch_api", self._fetch_api)
        workflow.add_node("synthesize", self._synthesize)

        workflow.set_entry_point("analyze")

        workflow.add_conditional_edges(
            "analyze",
            self._route_query,
            {
                "rag_only": "fetch_rag",
                "api_only": "fetch_api",
                "both": "fetch_rag"
            }
        )

        workflow.add_conditional_edges(
            "fetch_rag",
            lambda state: "fetch_api" if state.get("requires_api") else "synthesize"
        )

        workflow.add_edge("fetch_api", "synthesize")
        workflow.add_edge("synthesize", END)

        return workflow

    async def _analyze_query(self, state):
        """Analyze query requirements"""
        query = state["query"].lower()
        data_source = state["data_source"]

        if data_source == DataSource.PDF_ONLY:
            state["requires_rag"] = True
            state["requires_api"] = False
        elif data_source == DataSource.API_ONLY:
            state["requires_rag"] = False
            state["requires_api"] = True
        elif data_source == DataSource.BOTH:
            state["requires_rag"] = True
            state["requires_api"] = True
        else:  # AUTO
            # Simple detection logic
            api_keywords = ["price", "stock", "current", "today", "market"]
            rag_keywords = ["report", "annual", "risk", "strategy", "business"]

            has_api = any(keyword in query for keyword in api_keywords)
            has_rag = any(keyword in query for keyword in rag_keywords)

            if has_api and has_rag:
                state["requires_rag"] = True
                state["requires_api"] = True
            elif has_api:
                state["requires_rag"] = False
                state["requires_api"] = True
            else:
                state["requires_rag"] = True
                state["requires_api"] = False

        return state

    def _route_query(self, state):
        """Route based on requirements"""
        if state["requires_rag"] and state["requires_api"]:
            return "both"
        elif state["requires_rag"]:
            return "rag_only"
        else:
            return "api_only"

    async def _fetch_rag(self, state):
        """Fetch RAG data"""
        result = await self.rag_tool.query(state["query"])
        state["rag_result"] = result
        return state

    async def _fetch_api(self, state):
        """Fetch stock data"""
        # Extract symbol (simple logic)
        query = state["query"].upper()
        symbol = "NVDA" if "NVDA" in query or "NVIDIA" in query else "NVDA"

        result = await self.stock_tool.get_price(symbol)
        state["api_result"] = result
        return state

    async def _synthesize(self, state):
        """Synthesize final response"""
        parts = []

        # Add RAG results
        if state.get("rag_result") and state["rag_result"]["status"] == "success":
            parts.append(f"Document Analysis:\n{state['rag_result']['answer']}")

        # Add API results
        if state.get("api_result") and state["api_result"]["status"] == "success":
            api_data = state["api_result"]
            parts.append(f"Current Stock Data:\n"
                         f"Symbol: {api_data['symbol']}\n"
                         f"Price: ${api_data['price']}\n"
                         f"Change: {api_data['change']}")

        if not parts:
            state["final_answer"] = "I couldn't gather the necessary information to answer your question."
        elif len(parts) == 1:
            state["final_answer"] = parts[0]
        else:
            # Synthesize both sources
            context = "\n\n".join(parts)
            prompt = f"""Combine the following information to provide a comprehensive answer to: {state['query']}

{context}

Provide a clear, integrated response that combines insights from both sources."""

            response = await self.model.generate_content_async(prompt)
            state["final_answer"] = response.text

        return state

    async def load_document(self, pdf_path):
        """Load document"""
        return await self.rag_tool.load_document(pdf_path)

    async def query(self, query, data_source=DataSource.AUTO):
        """Main query method"""
        try:
            initial_state = {
                "query": query,
                "data_source": data_source,
                "rag_result": {},
                "api_result": {},
                "final_answer": "",
                "requires_rag": False,
                "requires_api": False,
                "error": ""
            }

            final_state = await self.app.ainvoke(initial_state)

            return {
                "status": "success",
                "answer": final_state["final_answer"],
                "sources": self._get_sources(final_state)
            }

        except Exception as e:
            logger.error(f"Agent query failed: {e}")
            return {"status": "error", "message": str(e)}

    def _get_sources(self, state):
        """Get sources used"""
        sources = []
        if state.get("rag_result") and state["rag_result"].get("status") == "success":
            sources.append("Annual Report")
        if state.get("api_result") and state["api_result"].get("status") == "success":
            sources.append("Stock API")
        return sources
