👤 Author
Abhishek Kumar Vishvakarma
Jr. Node.JS Developer | React | AI & RAG Learner

📄 AI RAG Pipeline with PDF using ChromaDB:
This project demonstrates a basic Retrieval-Augmented Generation (RAG) pipeline using PDF documents, sentence embeddings, and ChromaDB as a vector database.
It allows users to load documents and ask questions based only on the document content.

🚀 Features:
Load text from PDF files
Split text into chunks
Generate embeddings using Sentence Transformers
Store embeddings in ChromaDB
Retrieve relevant document chunks for user queries
Beginner-friendly and easy to understand

🧠 What is RAG?: 
RAG (Retrieval-Augmented Generation) is a technique where:
Documents are converted into embeddings
Stored in a vector database
User queries retrieve relevant document parts
Answers are generated only from retrieved content
📌 This project focuses on retrieval, which is the core of RAG.

ai_rag_project/
│
├── rag / ( rag_qa.py | rag.py | myfile.pdf | ai_notes.pdf )
├── docs.pdf
├── requirements.txt
├── app.py
└── .gitignore
|_rag_env
|_data/uploads/myfile.pdf

🛠️ Tech Stack:
Python 3.9+
ChromaDB
Sentence Transformers
LangChain Text Splitter
PyPDF

⚙️ Installation & Setup:
1️⃣ Clone the Repository:
git clone https://github.com/your-username/ai-rag-project.git
cd ai-rag-project


2️⃣ Create Virtual Environment:
python -m venv rag_env
rag_env\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

📄 Add Your PDF
Place your PDF file in the project folder
Update the file name inside rag.py:
pdf_path = "docs.pdf"   📌 Make sure the PDF is text-based, not scanned images.

▶️ Run the Project
python rag.py

🔍 Example Questions You Can Ask:
What is Artificial Intelligence?
Where is AI used?
Explain AI in simple words
What are applications of AI?
⚠️ Answers will only come from the PDF content.

📌 Important Notes
RAG does NOT generate answers on its own
If the answer is not in the document, it will not be returned
Larger and informative PDFs give better results


📦 requirements.txt:
pypdf
chromadb
sentence-transformers
langchain-text-splitters

⭐ Support
If you found this project helpful, please give it a ⭐ on GitHub!


__________________________________________________________________________________________________________


RAG Run by api 

command : ai_rag_project> rag_env\Scripts\activate
command : (rag_env) inside run this command > uvicorn app:app --reload (uvicorn <file_name>:<FastAPI_instance_name> --reload)
