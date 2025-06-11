from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_and_chunk_pptx_file(file_path, chunk_size=1000, chunk_overlap=200):
    """
    Load a PPTX file and chunk its text content using LangChain
    
    Args:
        file_path: Path to the PPTX file
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of LangChain Document objects
    """
    # Load the PPTX file using UnstructuredPowerPointLoader
    loader = UnstructuredPowerPointLoader(file_path)
    documents = loader.load()
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)
    
    return chunks

# pptx_file =  "ML.pptx" # Replace with your file path
    
# try:
#         chunks = load_and_chunk_pptx_file(pptx_file, chunk_size=500, chunk_overlap=100)
        
#         print(f"Successfully loaded and chunked {pptx_file}")
#         print(f"Created {len(chunks)} chunks")
        
#         # Display first few chunks
#         for i, chunk in enumerate(chunks[:3]):
#             print(f"\n--- Chunk {i+1} ---")
#             print(chunk.page_content[:200] + "...")
#             print(f"Metadata: {chunk.metadata}")
            
# except FileNotFoundError:
#         print(f"File {pptx_file} not found. Please check the file path.")
# except Exception as e:
#         print(f"Error processing file: {e}")