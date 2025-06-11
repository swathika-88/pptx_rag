from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from loading_chunking import load_and_chunk_pptx_file

def create_retriever(chunks, k=3):
    """Create a simple retriever without LLM"""
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    return retriever

# def retrieve_documents(retriever, query):
#     """Retrieve relevant documents for a query"""
#     docs = retriever.get_relevant_documents(query)
#     return docs


# chunks =load_and_chunk_pptx_file("ML.pptx")  # from previous PPTX loading code
# retriever = create_retriever(chunks)
# relevant_docs = retrieve_documents(retriever, "what is this presentation about?")
# for doc in relevant_docs:
#     print(doc.page_content)