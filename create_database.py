# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil

print("Imports completados correctamente.")


# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']
print(openai.api_key)

CHROMA_PATH = "chroma"
DATA_PATH = "data/"

print("Environment variables loaded correctly.")

print("Starting main...")
def main():
    print("Starting process...")
    try:
        generate_data_store()
    except Exception as e:
        print(f"Error in process: {str(e)}")

def generate_data_store():
    print("Generating data store...")
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)
    print("Data store generated successfully.")

def load_documents():
    print("Loading documents...")
    # Cargar archivos .txt
    txt_loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    txt_documents = txt_loader.load()

    # Cargar archivos .pdf
    pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()

    # Combinar todos los documentos
    documents = txt_documents + pdf_documents
    print(f"Loaded {len(documents)} documents.")
    return documents

def split_text(documents: list[Document]):
    print(f"Splitting {len(documents)} documents.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=700,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    print("Saving to Chroma...")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )

if __name__ == "__main__":
    main()