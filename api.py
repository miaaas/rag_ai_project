from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
import os

app = FastAPI(
    title="Document QA API",
    description="API for querying documents using RAG",
    version="1.0.0"
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="chroma_db", 
    embedding_function=embeddings
)

llm = Ollama(
    model="mistral",
    temperature=0.3
)

class Query(BaseModel):
    text: str

class Response(BaseModel):
    query: str
    answer: str
    context: list[str]

@app.post("/ask", response_model=Response)
async def ask_question(query: Query):
    try:
        retrieved_docs = vectorstore.similarity_search(query.text, k=4)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        
        # Prepare prompt
        prompt = f"""Based on the given context, answer the question.
        
        
        Context:
        {context}
        
        Question: {query}
        
        """
        
        # Get response from LLM
        response = llm.invoke(prompt)
        
        return Response(
            query=query.text,
            answer=response,
            context=[doc.page_content for doc in retrieved_docs]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/info")
async def get_info():
    return {
        "model": "mistral",
        "embeddings": "all-MiniLM-L6-v2",
        "vector_store": "chroma"
    }

# Run with: uvicorn api:app --reload
# Go to http://127.0.0.1:8000/docs to test the API