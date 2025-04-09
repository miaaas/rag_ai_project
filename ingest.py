from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

loader = TextLoader("document.txt")  
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(
    texts, 
    embeddings,
    persist_directory="chroma_db"  # persist_directory to save the database
)

print("Document successfully ingested into the database!")
