from sentence_transformers import SentenceTransformer

# Load a pretrained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Three sentences to compare
sentences = [
    "The patient shows symptoms of anemia and fatigue.",
    "Low red blood cell count can cause tiredness.",
    "The stock market rose sharply today."
]

# Convert each sentence into a vector
embeddings = model.encode(sentences)

# Show the shape of what we got back
print("Number of sentences:", len(sentences))
print("Shape of embeddings:", embeddings.shape)
print("First 10 numbers of sentence 1's vector:", embeddings[0][:10])

from sentence_transformers import util

# Compare sentence 1 with sentence 2, and sentence 1 with sentence 3
sim_1_2 = util.cos_sim(embeddings[0], embeddings[1])
sim_1_3 = util.cos_sim(embeddings[0], embeddings[2])

print("Similarity between sentence 1 and 2 (both medical):", sim_1_2.item())
print("Similarity between sentence 1 and 3 (medical vs stock market):", sim_1_3.item())