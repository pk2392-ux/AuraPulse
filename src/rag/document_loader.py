import os
import logging
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# We switch back to the standard HuggingFace embeddings which run locally
# and avoid Google API version mismatches!
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# 1. Setup Environment & Logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "data", "knowledge", "health_protocols.txt")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

def build_vector_database():
    """
    Reads the text file, splits it into chunks, converts them to vectors (embeddings),
    and saves them into a local ChromaDB database.
    """
    logger.info("Starting Vector Database build process...")

    # Step 1: Ensure the knowledge file exists
    if not os.path.exists(KNOWLEDGE_FILE):
        logger.error(f"Knowledge file not found at {KNOWLEDGE_FILE}")
        return

    # Step 2: Load the Document
    logger.info("Loading document into memory...")
    loader = TextLoader(KNOWLEDGE_FILE)
    documents = loader.load()

    # Step 3: Split the Text into Chunks
    logger.info("Splitting document into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # Max characters per chunk
        chunk_overlap=50    # How much chunks overlap (so we don't cut a sentence in half)
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split document into {len(chunks)} chunks.")

    # Step 4: Initialize the Embedding Model (LOCAL)
    # Using an open-source, local model is safer, free, and bypasses Google's API versioning issues.
    logger.info("Initializing Local HuggingFace Embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Step 5: Create and Persist the Vector Database
    logger.info(f"Building ChromaDB at {CHROMA_DB_DIR}...")
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    logger.info("✅ Vector Database successfully built and saved to disk!")

if __name__ == "__main__":
    build_vector_database()