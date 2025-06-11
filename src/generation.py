from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from retrieval import create_retriever
from loading_chunking import load_and_chunk_pptx_file
import os 
from dotenv import load_dotenv


load_dotenv()

API_KEY1 = os.getenv("API_KEY1")


def create_generator(retriever,api_key = API_KEY1,model_name= "llama3-8b-8192",custom_prompt= None,temperature = 0.1):
    """
    Create a RAG generator chain.
    
    Args:
        retriever: A retriever instance
        api_key: OpenAI API key
        model_name: Name of the LLM to use
        custom_prompt: Optional custom prompt template
        
    Returns:
        A RAG chain instance
    """
 # Initialize Groq LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature
    )
    
    # Create prompt template
    if custom_prompt:
        prompt = ChatPromptTemplate.from_template(custom_prompt)
    else:
        prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant. Use the following context to answer the question.
            If you don't know the answer based on the context, say "I don't have enough information to answer this question."
            
            Context:
            {context}
            
            Question:
            {input}
            """
        )
    
    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create retrieval chain
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain

chunks = load_and_chunk_pptx_file("ML.pptx")
retriever = create_retriever(chunks)

# Create the RAG generator
rag_generator = create_generator(retriever)

# Use the generator
response = rag_generator.invoke({"input": "What is the main topic discussed in the documents?"})
print(response["answer"])
