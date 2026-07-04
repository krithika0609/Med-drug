import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = np.load("chunk_embeddings.npy")
all_chunks = np.load("all_chunks.npy", allow_pickle=True)
chunk_sources = np.load("chunk_sources.npy", allow_pickle=True)

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

# Pick a handful of real questions straight from the dataset (easy test)
# We're just grabbing distinct source questions we've already seen work
exact_test_set = [
    "What are the symptoms of Iron-Deficiency Anemia ?",
    "What are the symptoms of diabetes ?",
]

# Now write the SAME questions in your own words (hard test)
rephrased_test_set = [
    "How do I know if I have low iron in my blood",
    "What does it feel like to have high blood sugar",
]
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
    
    return diverse_indices

def check_retrieval_diverse(question, expected_source_keyword, top_k=3, candidate_pool=10):
    diverse_indices = get_diverse_indices(question, top_k, candidate_pool)
    retrieved_sources = [chunk_sources[i] for i in diverse_indices]
    found = any(expected_source_keyword.lower() in src.lower() for src in retrieved_sources)
    return found, retrieved_sources


# Test the exact questions first
print("=== EXACT QUESTION TEST ===")
for q in exact_test_set:
    found, sources = check_retrieval_diverse(q, expected_source_keyword=q.split("?")[0][-20:])
    print(f"Q: {q}")
    print(f"Found relevant source: {found}")
    print(f"Top sources: {sources}\n")
# Now the harder test — rephrased questions, different wording entirely
print("=== REPHRASED QUESTION TEST ===")
rephrased_checks = [
    ("How do I know if I have low iron in my blood", "Iron-Deficiency Anemia"),
    ("What does it feel like to have high blood sugar", "Diabetes"),
]

for question, expected_keyword in rephrased_checks:
    found, sources = check_retrieval_diverse(question, expected_source_keyword=expected_keyword)
    print(f"Q: {question}")
    print(f"Found relevant source: {found}")
    print(f"Top sources: {sources}\n")
# Aggregate all tests into one accuracy score
all_tests = [
    ("What are the symptoms of Iron-Deficiency Anemia ?", "Iron-Deficiency Anemia"),
    ("What are the symptoms of diabetes ?", "Diabetes"),
    ("How do I know if I have low iron in my blood", "Iron-Deficiency Anemia"),
    ("What does it feel like to have high blood sugar", "Diabetes"),
]

correct = 0
for question, expected_keyword in all_tests:
    found, _ = check_retrieval_diverse(question, expected_source_keyword=expected_keyword)
    if found:
        correct += 1

accuracy = correct / len(all_tests) * 100
print(f"\nRetrieval Accuracy@3: {accuracy:.1f}% ({correct}/{len(all_tests)})")

import random

random.seed(42)  # ensures we get the same "random" sample every time we run this

# Automatically build a larger test set by sampling real questions from the dataset
# We test: does searching for the ORIGINAL question retrieve its OWN chunk back?
sample_size = 100
sample_indices = random.sample(range(len(all_chunks)), sample_size)

correct = 0
for idx in sample_indices:
    question = chunk_sources[idx]
    diverse_indices = get_diverse_indices(question, top_k=3, candidate_pool=10)
    if idx in diverse_indices:
        correct += 1

accuracy = correct / sample_size * 100
print(f"\nDIVERSE Retrieval Accuracy@3 on {sample_size} random samples: {accuracy:.1f}% ({correct}/{sample_size})")


# Let's look at a few actual failures to understand what's happening
print("\n=== INSPECTING FAILURES ===")
shown = 0
for idx in sample_indices:
    if shown >= 3:
        break
    question = chunk_sources[idx]
    query_embedding = embed_model.encode([question])
    distances, indices = index.search(query_embedding, 3)
    
    if idx not in indices[0]:
        print(f"\nOriginal question: {question}")
        print(f"Expected chunk: {all_chunks[idx][:150]}")
        print("Top 3 retrieved instead:")
        for i in indices[0]:
            print(f"  - [{chunk_sources[i]}]: {all_chunks[i][:100]}")
        shown += 1
from sentence_transformers import util

def relaxed_check(idx, top_k=3, candidate_pool=10, similarity_threshold=0.6):
    question = chunk_sources[idx]
    diverse_indices = get_diverse_indices(question, top_k, candidate_pool)
    
    expected_embedding = chunk_embeddings[idx]
    
    best_similarity = 0
    for retrieved_idx in diverse_indices:
        sim = util.cos_sim(expected_embedding, chunk_embeddings[retrieved_idx]).item()
        best_similarity = max(best_similarity, sim)
    
    return best_similarity >= similarity_threshold, best_similarity

# Re-run the 100-sample test with the relaxed, content-based metric
correct = 0
for idx in sample_indices:
    is_correct, sim = relaxed_check(idx)
    if is_correct:
        correct += 1

accuracy = correct / sample_size * 100
print(f"\nRELAXED (content-similarity) Accuracy@3 on {sample_size} samples: {accuracy:.1f}% ({correct}/{sample_size})")