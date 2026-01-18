from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import os
import uuid

app = FastAPI()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection("rag_collection")


def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = extract_text_from_pdf(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)
    embeddings = model.encode(chunks)

    ids = [str(uuid.uuid4()) for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=ids
    )

    return {
        "message": "PDF processed successfully",
        "chunks": len(chunks)
    }


@app.post("/ask")
async def ask_question(question: str):
    query_embedding = model.encode([question])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3
    )

    return {
        "question": question,
        "answers": results["documents"][0]
    }
