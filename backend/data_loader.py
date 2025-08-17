import os
import faiss
import pickle
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = "../data"
INDEX_FILE = "college_index.faiss"
META_FILE = "college_meta.pkl"

# Lightweight embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_text_files():
    texts = []
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
        elif file.endswith(".pdf"):
            pdf = PdfReader(path)
            for page in pdf.pages:
                if page.extract_text():
                    texts.append(page.extract_text())
    return texts

def chunk_text(text, chunk_size=300):
    """Break long text into smaller chunks."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

def build_index():
    print("📄 Loading college data...")
    raw_texts = load_text_files()

    chunks = []
    for doc in raw_texts:
        for chunk in chunk_text(doc):
            chunks.append(chunk)

    print(f"✅ Total chunks: {len(chunks)}")

    # Create embeddings
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save metadata & index
    with open(META_FILE, "wb") as f:
        pickle.dump(chunks, f)

    faiss.write_index(index, INDEX_FILE)
    print("🎯 Index built and saved.")

if __name__ == "__main__":
    build_index()
