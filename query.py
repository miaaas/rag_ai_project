from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
import os

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="chroma_db", 
    embedding_function=embeddings
)

llm = Ollama(
    model="mistral", 
    temperature=0.3
)

def get_response(query: str):
    retrieved_docs = vectorstore.similarity_search(query, k=4)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt = f"""Based on the given context, answer the question.
    
    Context:
    {context}
    
    Question: {query}
    
    """
    
    response = llm.invoke(prompt)
    return response

if __name__ == "__main__":
    print("\n💡 Welcome! Ask a question about your document.")
    while True:
        query = input("\nYour question (or 'exit' for exit): ")
        if query.lower() in ['exit', 'quit', 'q']:
            break
        print("\nAnswer:", get_response(query))
