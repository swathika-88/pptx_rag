from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from loading_chunking import load_and_chunk_pptx_file

def create_vector_store(chunks, persist_directory="./chroma_db"):
    """Create ChromaDB vector store using LangChain"""
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vectorstore

def search(vectorstore, query, k=3):
    """Search the vector store"""
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

# Load and create vector store
chunks = load_and_chunk_pptx_file("ML.pptx")
vectorstore = create_vector_store(chunks)
    
# Search
results = search(vectorstore, "What is supervised learning?")
for i, doc in enumerate(results):
        print(f"{i+1}. {doc[:100]}...")