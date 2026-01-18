from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb


def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# pdf_path = "myfile.pdf"
pdf_path = "ai_notes.pdf"
text = extract_text_from_pdf(pdf_path)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(chunks).tolist()

client = chromadb.Client()
collection = client.create_collection(name="rag_collection")

collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[str(i) for i in range(len(chunks))]
)

qa_model = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_new_tokens=200
)

while True:
    query = input("\nAsk question (type exit to quit): ")
    if query.lower() == "exit":
        break

    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    context = " ".join(results["documents"][0])

    prompt = f"""
    Answer the question using only the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = qa_model(prompt)
    print("\nAnswer:\n", response[0]["generated_text"])
