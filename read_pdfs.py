import os
from pathlib import Path
from dotenv import load_dotenv


# langchain_community faiss-cpu langchain-text-splitters pymupdf
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

docs = []
for pdf_path in Path("pdfs").glob("*.pdf"):
    try:
        loader = PyMuPDFLoader(str(pdf_path))
        docs.extend(loader.load())
        print(f"-----Carregado {pdf_path.name}-----")
    except Exception as e:
        print(f"Erro ao carregar {pdf_path}: {e}")

splitters = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitters.split_documents(docs)

GOOGLE_IA_KEY = os.getenv("GOOGLE_IA_KEY")
print("Usando chave de API do Google:", "SIM" if GOOGLE_IA_KEY else "NÃO")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_IA_KEY
)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.3, "k": 4}
)

prompt_rag = ChatPromptTemplate.from_template(
    "Você é um assistente de Políticas Internas (RH/TI) da empresa Carraro Desenvolvimento. "
    "Responda somente com base nos documentos fornecidos. "
    "Se não houver base suficiente, responda 'Não sei'.\n\n"
    "Documentos:\n{context}\n\nPergunta: {input}\n"
)

