import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import chromadb

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        # text += page.extract_text()
        # text += page.extract_text().replace("\n", " ")
        text += re.sub(r"\s+", " ", page.extract_text())
    return text

pdf_path = "myfile.pdf"
pdf_text = extract_text_from_pdf(pdf_path)
print(pdf_text)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_text(pdf_text)

print(len(chunks))
print(chunks[0])

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

client = chromadb.Client()

collection = client.create_collection("pdf_rag")

collection.add(
    documents=chunks,
    embeddings=list(embeddings),
    ids=[str(i) for i in range(len(chunks))]
)

user_query = "What is the main idea discussed in the document?"
query_embedding = model.encode([user_query])

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

for doc in results["documents"][0]:
    print(doc)
    print("-----")
