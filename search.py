import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load what we saved earlier
chunk_embeddings = np.load("chunk_embeddings.npy")
all_chunks = np.load("all_chunks.npy", allow_pickle=True)
chunk_sources = np.load("chunk_sources.npy", allow_pickle=True)

print("Loaded", len(all_chunks), "chunks with embeddings shape", chunk_embeddings.shape)

# Build the FAISS index
dimension = chunk_embeddings.shape[1]  # 384
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

print("Index built. Total vectors in index:", index.ntotal)

model = SentenceTransformer('all-MiniLM-L6-v2')

def search(query, top_k=3):
    # Embed the query the same way we embedded the chunks
    query_embedding = model.encode([query])
    
    # Search the index for the closest chunks
    distances, indices = index.search(query_embedding, top_k)
    
    print(f"\nQuery: {query}")
    print("-" * 50)
    for rank, idx in enumerate(indices[0]):
        print(f"\nResult {rank + 1} (distance: {distances[0][rank]:.4f})")
        print("Source question:", chunk_sources[idx])
        print("Chunk:", all_chunks[idx][:300])

# Try it
search("What are the symptoms of anemia?")