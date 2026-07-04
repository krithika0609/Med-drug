from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # your React app's address
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load everything once when the server starts (not on every request)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = np.load("chunk_embeddings.npy")
all_chunks = np.load("all_chunks.npy", allow_pickle=True)
chunk_sources = np.load("chunk_sources.npy", allow_pickle=True)

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

# Define the shape of an incoming request
class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "Medical RAG API is running"}
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_diverse_indices(question, top_k=3, candidate_pool=10):
    query_embedding = embed_model.encode([question])
    distances, indices = index.search(query_embedding, candidate_pool)
    
    seen_sources = set()
    diverse_indices = []
    for idx in indices[0]:
        source = chunk_sources[idx]
        if source not in seen_sources:
            seen_sources.add(source)
            diverse_indices.append(idx)
        if len(diverse_indices) >= top_k:
            break
    return diverse_indices, distances[0][0]

@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question
    diverse_indices, best_distance = get_diverse_indices(question)
    
    if best_distance > 1.0:
        return {
            "answer": "I don't have relevant information to answer that question confidently.",
            "sources": []
        }
    
    retrieved_chunks = [all_chunks[i] for i in diverse_indices]
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Question: {question}\n\nAnswer the question using only the context below.\n\nContext:\n{context}\n\nAnswer:"
    
    answer = generate_answer(prompt)
    
    return {
        "answer": answer,
        "sources": [chunk_sources[i] for i in diverse_indices]
    }