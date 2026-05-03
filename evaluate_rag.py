import os
import pandas as pd
import asyncio
from sqlalchemy import create_engine
import chromadb
import nest_asyncio

# Apply async fix for environments that need it
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

# Hybrid Search Imports
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

# Evaluation Imports
from llama_index.core.evaluation import CorrectnessEvaluator

def setup_agent():
    """Initializes the Agent and Tools (Identical to your app.py)"""
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    # Using 70B for both the Agent and the Judge
    #llm = Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key, temperature=0.0)
    llm = Groq(model="openai/gpt-oss-120b", api_key=groq_api_key, temperature=0.0)
    Settings.llm = llm
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    # 1. SQL Tool
    engine = create_engine("sqlite:///claims_database.db")
    sql_database = SQLDatabase(engine, include_tables=["policies", "claims", "claim_history"])
    sql_query_engine = NLSQLTableQueryEngine(sql_database=sql_database, tables=["policies", "claims", "claim_history"])
    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        name="sql_database_qa_tool",
        description="Use this tool to ask questions about structured data. Always double escape apostrophes."
    )

    # 2. Hybrid Tool
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
        nodes = hybrid_retriever.retrieve(query)
        if not nodes: return "No documents found."
        return "\n\n".join([node.node.get_content() for node in nodes])

    vector_tool = FunctionTool.from_defaults(fn=true_hybrid_search, name="ocr_document_tool")

    # 3. Agent
    agent = FunctionAgent(
        name="Claims_Agent",
        tools=[sql_tool, vector_tool],
        llm=llm,
        system_prompt="You are an expert claims assistant. Use your tools to find facts. Do not narrate your steps."
    )
    
    return agent, llm

async def run_evaluation():
    print("\n[SYSTEM] Initializing Agent and Evaluator...")
    agent, judge_llm = setup_agent()
    
    # Initialize the LLM-as-a-Judge
    evaluator = CorrectnessEvaluator(llm=judge_llm)
    
    # Load the dataset
    print("[SYSTEM] Loading eval_dataset.csv...")
    df = pd.read_csv("eval_dataset.csv")
    
    results = []

    print("\n" + "="*50)
    print("🚀 STARTING AUTOMATED EVALUATION")
    print("="*50)

    for index, row in df.iterrows():
        question = row['question']
        expected_answer = row['expected_answer']
        
        print(f"\nEvaluating Question {index + 1}/{len(df)}: {question}")
        
        try:
            # 1. Ask the Agent
            ctx = Context(agent)
            response = await agent.run(user_msg=question, ctx=ctx)
            actual_answer = str(response)
            
            # 2. Judge the Answer
            eval_result = await evaluator.aevaluate(
                query=question,
                response=actual_answer,
                reference=expected_answer
            )
            
            print(f"Score: {eval_result.score}/5.0")
            print(f"Feedback: {eval_result.feedback}")
            
            score = eval_result.score
            feedback = eval_result.feedback

        except Exception as e:
            # THE SAFETY NET: Catches the Max Iterations error!
            print(f"❌ Agent crashed on this question: {e}")
            actual_answer = "ERROR: Agent got stuck in a loop or rate limited."
            score = 0.0
            feedback = f"Agent failed to complete the task. Error: {e}"
        
        # 3. Save to results list
        results.append({
            "Question": question,
            "Expected": expected_answer,
            "Actual": actual_answer,
            "Score": score,
            "Feedback": feedback
        })
        
        # Add a tiny pause to respect Groq's free-tier rate limits!
        await asyncio.sleep(2)

    # Save final report
    results_df = pd.DataFrame(results)
    results_df.to_csv("evaluation_report.csv", index=False, encoding="utf-8-sig")
    print("\n" + "="*50)
    print(f"✅ Evaluation Complete! Average Score: {results_df['Score'].mean()}/5.0")
    print("Report saved to 'evaluation_report.csv'")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(run_evaluation())