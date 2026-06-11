import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

def get_relevant_context(user_question: str, k: int = 2) -> str:
    """
    Takes a user's question, searches the local ChromaDB vector store,
    and returns the most relevant paragraphs as a single joined string.
    
    k: The number of chunks to retrieve (default is top 2).
    """
    if not os.path.exists(CHROMA_DB_DIR):
        logger.warning("ChromaDB directory not found. Returning empty context.")
        return ""

    try:
        # 1. Re-initialize the EXACT same embedding model used during ingestion
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. Connect to the existing Chroma Database
        db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        
        # 3. Perform the Similarity Search
        # This converts the user_question to a vector, finds the closest 'k' vectors in the DB, 
        # and returns the original text documents attached to those vectors.
        results = db.similarity_search(user_question, k=k)
        
        # 4. Format the Results
        # results is a list of Document objects. We extract the page_content (the text)
        # and join them together with double newlines so Gemini can read them easily.
        context_string = "\n\n".join([doc.page_content for doc in results])
        
        return context_string

    except Exception as e:
        logger.error(f"Failed to retrieve context from Vector DB: {e}")
        return ""

# Simple test block: if you run this file directly, it will test the search!
if __name__ == "__main__":
    test_question = "What should I do if my heart rate is elevated?"
    print(f"Question: {test_question}\n")
    print("--- RETRIEVED CONTEXT ---")
    print(get_relevant_context(test_question))
    print("-------------------------")