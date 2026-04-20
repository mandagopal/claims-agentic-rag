import chromadb
import re
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter 

# Use local Ollama for embeddings
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

def extract_exact_metadata(text):
    """Scan text for exact Claim IDs and Policy Numbers using Regex."""
    metadata = {}
    
    # Matches patterns like CLM-00788 or CLM00788
    claim_match = re.search(r'CLM-?\d{5}', text, re.IGNORECASE)
    if claim_match:
        # Standardize format to CLM00788 to ensure perfect matching
        metadata['claim_id'] = claim_match.group(0).upper().replace('-', '')
    
    # Matches patterns like POL-12345 or POL-WC-558812
    policy_match = re.search(r'POL-?[A-Z]*?-?\d+', text, re.IGNORECASE)
    if policy_match:
        metadata['policy_no'] = policy_match.group(0).upper()
        
    return metadata

def ingest_documents():
    print("Reading text documents...")
    documents = SimpleDirectoryReader("./data/txt").load_data()
    
    # Split into 256-token chunks
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    
    print("Extracting metadata and formatting for citations...")
    for node in nodes:
        # 1. Dynamically extract claim and policy numbers
        extracted_meta = extract_exact_metadata(node.get_content())
        node.metadata.update(extracted_meta)
        
        # 2. Format for LLM citations
        node.metadata_separator = ":: "
        node.metadata_template = "{key}: {value}"
        node.text_template = "Metadata:\n{metadata_str}\n-----\nContent:\n{content}"

    print("Initializing ChromaDB...")
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("claims_ocr")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("Embedding and indexing documents...")
    # Pass the processed nodes directly into the index
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    print("Vector database created successfully at ./chroma_db")

if __name__ == "__main__":
    ingest_documents()
