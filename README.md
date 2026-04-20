# 🛡️ Claims Agentic RAG: Multi-Route Hybrid Retrieval System

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10+-black.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-orange.svg)

An enterprise-grade, Proof-of-Concept (POC) Agentic RAG application designed to unify siloed structured and unstructured data. This project demonstrates how to build an intelligent routing agent capable of writing complex SQL queries against relational databases and performing advanced Hybrid Search over OCR text narratives.

## 🧠 The Problem & The Solution

In industries like Insurance, Legal, and Finance, critical data is divided:
* **Structured Data (SQL):** Claim statuses, financial reserves, dates.
* **Unstructured Data (OCR/Text):** Police reports, claimant narratives, incident descriptions.

Standard Vector Search models suffer from the **"Vector Blindspot"**—they understand semantic meaning but fail to retrieve exact alphanumeric IDs (e.g., `CLM-00788`). 

**The Solution:** This project implements an **Agentic Router** that intercepts user queries and routes them to specialized engines. Unstructured queries utilize **True Hybrid Search**, combining Dense Vector embeddings (meaning) with Sparse BM25 retrieval (exact keywords) and merging them using **Reciprocal Rank Fusion (RRF)** for 100% retrieval accuracy on exact IDs.

---

## 🏗️ System Architecture

![Agentic RAG Architecture](claims_rag_routing_architecture.svg)


1. **User Prompt** ➡️ **Llama 3.3 70B Routing Agent**
2. **Routing Agent** branches into two paths:
   * **Path A (Structured):** Natural Language to SQL Engine ➡️ queries SQLite Database. Includes custom logic to double-escape apostrophes and prevent SQL injection/syntax crashes.
   * **Path B (Unstructured):** Hybrid Retriever ➡️ searches ChromaDB (Vectors) + BM25 (Keywords) ➡️ Merges at Reciprocal Rank Fusion (RRF).
3. **Agent Synthesizer** outputs the final answer with precise document citations.

---

## 🛠️ Tech Stack

* **Orchestration / Agent:** LlamaIndex & Llama 3.3 70B (via Groq API)
* **Embedding Model:** Ollama (`nomic-embed-text` running locally)
* **Vector Database:** ChromaDB
* **Sparse Retrieval:** BM25
* **Relational Database:** SQLite
* **User Interface:** Streamlit

---

## 📂 Repository Structure

```text
claims-agentic-rag/
│
├── data/
│   └── txt/                     # Raw OCR text documents (dummy data)
│
├── app.py                       # Main Streamlit UI & Agentic Workflow
├── ingest_vector.py             # ETL script: chunks documents and builds ChromaDB & BM25 indices
├── ingest_csv_into_sqlite.py    # ETL script: builds the structured SQLite database
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignored files (venv, DBs, etc.)
└── README.md                    # Project documentation

🚀 Installation & Setup
1. Clone the Repository

git clone [https://github.com/your-username/claims-agentic-rag.git](https://github.com/your-username/claims-agentic-rag.git)
cd claims-agentic-rag

2. Create and Activate Virtual Environment

python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

4. Set Environment Variables
You will need a free API key from Groq.


export GROQ_API_KEY="your_groq_api_key_here"

5. Build the Databases
Ensure Ollama is running locally with the nomic-embed-text model (ollama run nomic-embed-text).


python3 ingest_csv_into_sqlite.py  # Builds the structured DB
python3 ingest_vector.py           # Builds the ChromaDB vector store

6. Launch the Application

streamlit run app.py

💡 Usage Examples
Once the Streamlit interface is running, try asking complex queries that test both databases:

"What is the total paid amount of all claims?" (Triggers Text-to-SQL)

"What is the INSURED'S PRELIMINARY RESPONSE provided by Derek Greenfield for the CLAIM #: CLM-00788?" (Triggers Hybrid Vector + BM25 Search)

🔮 Future Enhancements (Roadmap)
GraphRAG: Implementing Knowledge Graphs (Neo4j) to map multi-hop relationships (e.g., linking frequent claimants to specific medical providers to detect fraud rings).

LLMOps Evaluation: Integrating RAGAS to automatically score context precision, recall, and faithfulness.

Enterprise Security: Adding Metadata Filtering for Role-Based Access Control (RBAC) and NeMo Guardrails for PII redaction.