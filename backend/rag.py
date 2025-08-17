import faiss
import pickle
from sentence_transformers import SentenceTransformer

INDEX_FILE = "college_index.faiss"
META_FILE = "college_meta.pkl"

model = SentenceTransformer("all-MiniLM-L6-v2")

def search_college_data(query, top_k=3):
    # Load index and metadata
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        chunks = pickle.load(f)

    # Encode query
    query_embedding = model.encode([query])

    # Search in FAISS
    distances, indices = index.search(query_embedding, top_k)

    results = [chunks[i] for i in indices[0]]
    return results

if __name__ == "__main__":
    while True:
        q = input("Ask about college: ")
        if q.lower() in ["exit", "quit"]:
            break
        answers = search_college_data(q)
        print("\nTop results:")
        for ans in answers:
            print("-", ans)
