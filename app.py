import streamlit as st
import asyncio
import nest_asyncio
import os
import re
from sqlalchemy import create_engine
import chromadb
import logging
import sys
import tiktoken

# --- TERMINAL LOGGING FIX ---
from llama_index.core import set_global_handler
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("llama_index").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
set_global_handler("simple")

# --- ASYNC FIX FOR STREAMLIT ---
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
nest_asyncio.apply()

from llama_index.core import SQLDatabase, VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq

# --- HYBRID SEARCH IMPORTS ---
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

# --- NEW: TOKEN TRACKING IMPORTS ---
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

@st.cache_resource
def initialize_tools():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("⚠️ GROQ_API_KEY environment variable is not set. Please export it in your terminal.")
        st.stop()

    # ==========================================
    # 0. LLMOps: TOKEN TRACKING SETUP
    # ==========================================
    # Initialize the token counter using a standard tokenizer
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model("gpt-4o").encode
    )
    # Attach the counter globally so it listens to every Agent action
    Settings.callback_manager = CallbackManager([token_counter])

    Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key, temperature=0.0)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    # ==========================================
    # 1. SQL ENGINE SETUP (Structured Data)
    # ==========================================
    engine = create_engine("sqlite:///claims_database.db")
    sql_database = SQLDatabase(engine, include_tables=["policies", "claims", "claim_history"])
    
    context_dict = {
        "policies": "Table of policies. Columns: 'policy_no', 'policyholder_name', 'dob', 'status', 'policy_type'.",
        "claims": "Table of claims. Columns: 'claim_id', 'policy_no', 'claimant_name', 'status', 'reserve_amount', 'paid_amount', 'description'.",
        "claim_history": "Table of claim events. Columns: 'event_id', 'claim_id', 'policy_no', 'event_type', 'notes'."
    }

    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["policies", "claims", "claim_history"],
        context_dict=context_dict,
        synthesize_response=False 
    )

    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        name="sql_database_qa_tool",
        description=(
            "Use this tool to ask questions about structured data (policies, claims, history). "
            "CRITICAL: Input MUST be a natural language question. DO NOT pass SQL code. "
            "CRITICAL SQL RULE: If the user's prompt contains an apostrophe, you MUST double escape it."
        )
    )

    # ==========================================
    # 2. TRUE HYBRID SEARCH ENGINE (Unstructured)
    # ==========================================
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("claims_ocr")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    vector_index = VectorStoreIndex.from_vector_store(vector_store)
    vector_retriever = vector_index.as_retriever(similarity_top_k=4)

    documents = SimpleDirectoryReader("./data/txt").load_data()
    for doc in documents:
        doc.metadata_separator = ":: "
        doc.metadata_template = "{key}: {value}"
        doc.text_template = "Metadata:\n{metadata_str}\n-----\nContent:\n{content}"
        
    nodes = Settings.node_parser.get_nodes_from_documents(documents)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=4)

    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=4,
        num_queries=1,
        mode="reciprocal_rerank",
    )

    def true_hybrid_search(query: str) -> str:
        """Searches unstructured claim forms, text documents, and OCR narratives."""
        print(f"\n[SYSTEM] Executing Hybrid Search (BM25 + Vector) for: '{query}'")
        nodes = hybrid_retriever.retrieve(query)
        if not nodes:
            return "No matching documents found in the database."
        return "\n\n".join([node.node.get_content() for node in nodes])

    vector_tool = FunctionTool.from_defaults(fn=true_hybrid_search, name="ocr_document_tool")

    # Return the token_counter alongside the tools so the UI can read it
    return [sql_tool, vector_tool], token_counter

# ==========================================
# 3. STREAMLIT UI & AGENT ORCHESTRATION
# ==========================================
st.set_page_config(page_title="Claims AI Assistant", layout="wide")
st.title("🛡️ Claims AI Assistant (Hybrid Search + 70B)")

if "agent" not in st.session_state:
    # Unpack the tools and the token counter
    tools, token_counter = initialize_tools()
    st.session_state.token_counter = token_counter
    
    smart_prompt = (
        "You are an expert, fact-based insurance claims assistant. "
        "CRITICAL RULES: "
        "1. CONTEXT AWARENESS: Before using any tool, review the chat history. Replace pronouns with actual names before searching. "
        "2. TOOL AFFINITY: Prioritize the ocr_document_tool for follow-ups about entities previously found in OCR. "
        "3. Never write SQL yourself. Pass plain English questions to the sql_database_qa_tool. "
        "4. DO NOT narrate your steps. "
        "5. If the sql_database_qa_tool returns an empty result or a SQL syntax error, SILENTLY and IMMEDIATELY trigger the ocr_document_tool. "
        "6. CITATIONS REQUIRED: When you answer using information from the ocr_document_tool, "
        "you MUST append the 'file_name' found in the Metadata block to the end of your answer. "
        "7. Only if BOTH tools return no results should you say: 'I cannot find this information.'"
    )
    
    st.session_state.agent = FunctionAgent(
        name="Claims_Routing_Agent",
        tools=tools,
        llm=Settings.llm,
        system_prompt=smart_prompt
    )
    st.session_state.ctx = Context(st.session_state.agent)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Create a clean sidebar to display the static Groq limits
with st.sidebar:
    st.header("⚙️ Groq API Limits")
    st.markdown("**Model:** `llama-3.3-70b-versatile`")
    st.markdown("**Free Tier TPM Limit:** ~6,000 Tokens/Min")
    st.caption("If your 'Total Tokens' in a single request approaches 6,000, you will hit a rate limit error (429 Too Many Requests). Wait 60 seconds for the limit to clear.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about claims, policies, or specific incidents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and searching databases..."):
            async def get_agent_response():
                return await st.session_state.agent.run(
                    user_msg=prompt, 
                    ctx=st.session_state.ctx
                )
            
            try:
                # Reset the counter right before the run to only count this specific turn
                st.session_state.token_counter.reset_counts()
                
                response = asyncio.run(get_agent_response())
                st.markdown(str(response))
                st.session_state.messages.append({"role": "assistant", "content": str(response)})
                
                # --- NEW: DISPLAY TELEMETRY DASHBOARD ---
                prompt_tokens = st.session_state.token_counter.prompt_llm_token_count
                completion_tokens = st.session_state.token_counter.completion_llm_token_count
                total_tokens = st.session_state.token_counter.total_llm_token_count
                
                with st.expander("📊 LLM Token Telemetry (Current Request)", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    col1.metric(label="Input (Prompt) Tokens", value=f"{prompt_tokens:,}")
                    col2.metric(label="Output (Completion) Tokens", value=f"{completion_tokens:,}")
                    col3.metric(label="Total Tokens Used", value=f"{total_tokens:,}")
                    
                    # Add a visual warning if you are approaching the Groq TPM limit
                    if total_tokens > 4500:
                        st.warning("⚠️ High Token Usage: You are approaching Groq's 6,000 TPM limit. You may need to wait 60 seconds before asking the next question to avoid a rate-limit error.")

            except Exception as e:
                st.error(f"⚠️ **Connection Timeout or Error:** {str(e)}")
