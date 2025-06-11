from sentence_transformers import SentenceTransformer
from loading_chunking import load_and_chunk_pptx_file

def create_embeddings(chunks = load_and_chunk_pptx_file, model_name='all-MiniLM-L6-v2'):
    """
    Create embeddings for document chunks using Sentence Transformers
    
    Args:
        chunks: List of LangChain Document objects
        model_name: Name of the Sentence Transformer model to use
    
    Returns:
        tuple: (embeddings_array, chunk_texts, chunk_metadata)
    """
    # Initialize the sentence transformer model
    model = SentenceTransformer(model_name)
    
    # Extract text content from chunks
    chunk_texts = [chunk.page_content for chunk in chunks]
    
    
    # Create embeddings
    print(f"Creating embeddings for {len(chunk_texts)} chunks...")
    embeddings = model.encode(chunk_texts, show_progress_bar=True)
    
    return embeddings