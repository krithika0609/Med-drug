from datasets import load_dataset

# Load the MedQuAD dataset from HuggingFace
dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset")

# Look at its structure
print(dataset)

# Look at the first 3 rows to understand the actual content
for i in range(3):
    print("Type:", dataset["train"][i]["qtype"])
    print("Question:", dataset["train"][i]["Question"])
    print("Answer:", dataset["train"][i]["Answer"][:200])  # first 200 chars only
    print("---")

# Check answer lengths across the dataset
lengths = [len(dataset["train"][i]["Answer"].split()) for i in range(len(dataset["train"]))]

print("Shortest answer (words):", min(lengths))
print("Longest answer (words):", max(lengths))
print("Average answer (words):", sum(lengths) / len(lengths))