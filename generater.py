from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the model and its tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

def generate_answer(prompt):
    # Convert text into tokens the model understands
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # Generate a response
    outputs = model.generate(**inputs, max_new_tokens=150)
    
    # Convert tokens back into readable text
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Quick test — no retrieval yet, just confirm the model works at all
test = generate_answer("What is the capital of France?")
print("Test answer:", test)
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load the same index setup from search.py
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = np.load("chunk_embeddings.npy")
all_chunks = np.load("all_chunks.npy", allow_pickle=True)
chunk_sources = np.load("chunk_sources.npy", allow_pickle=True)

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)
def rag_answer(question, top_k=2, distance_threshold=1.0):
    query_embedding = embed_model.encode([question])
    distances, indices = index.search(query_embedding, top_k)
    best_distance = distances[0][0]
    
    # If the best match isn't actually close, refuse rather than guess
    if best_distance > distance_threshold:
        return "I don't have relevant information to answer that question confidently.", [], best_distance
    
    retrieved_chunks = [all_chunks[idx] for idx in indices[0]]
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Question: {question}\n\nAnswer the question using only the context below.\n\nContext:\n{context}\n\nAnswer:"
    answer = generate_answer(prompt)
    return answer, retrieved_chunks, best_distance
# Interactive loop — ask questions until you type 'quit'
print("\nMedical RAG Assistant (type 'quit' to exit)")
while True:
    question = input("\nYour question: ")
    if question.lower() == "quit":
        break
    answer, chunks, best_distance = rag_answer(question)
    print("Answer:", answer)
    print("\nRetrieved chunks were:")
    for c in chunks:
        print("-", c[:150])