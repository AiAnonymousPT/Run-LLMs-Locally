import os
import shutil
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Function to ensure directory exists and is empty
def ensure_empty_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# Ensure vector-db directory exists and is empty
persist_directory = "./vector-db"
ensure_empty_directory(persist_directory)

# Load documents from directory
path = "./docs"
loader = DirectoryLoader(path)
docs = loader.load()
print('Docs loaded')

# Text split
text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
documents = text_splitter.split_documents(docs)
print('Docs split')

# Generate and store embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
print('Vector db created')