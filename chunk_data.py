from datasets import load_dataset

dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset")
data = dataset["train"]

# Filter out rows with very short answers (likely broken data)
filtered = [row for row in data if len(row["Answer"].split()) >= 5]

print("Original rows:", len(data))
print("After filtering short answers:", len(filtered))

def chunk_text(text, max_words=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks
    # Apply chunking to every filtered row, and track metadata for each chunk
all_chunks = []       # the actual text of each chunk
chunk_sources = []    # which original question each chunk came from

for row in filtered:
    pieces = chunk_text(row["Answer"], max_words=300)
    for piece in pieces:
        all_chunks.append(piece)
        chunk_sources.append(row["Question"])

print("Total chunks created:", len(all_chunks))
print("\nExample chunk 0:")
print(all_chunks[0])
print("\nCame from question:", chunk_sources[0])

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

print("Embedding", len(all_chunks), "chunks... this will take a few minutes.")
chunk_embeddings = model.encode(all_chunks, show_progress_bar=True)

# Save everything to disk so we don't have to redo this every time
np.save("chunk_embeddings.npy", chunk_embeddings)
np.save("all_chunks.npy", np.array(all_chunks, dtype=object))
np.save("chunk_sources.npy", np.array(chunk_sources, dtype=object))

print("Saved. Embeddings shape:", chunk_embeddings.shape)