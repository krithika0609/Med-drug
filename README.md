# MedRAG — AI Medical Q&A Assistant

A full-stack Retrieval-Augmented Generation (RAG) system that answers medical
questions by retrieving relevant passages from a trusted knowledge base and
generating grounded, source-cited answers — with a built-in safeguard against
answering confidently when no relevant information exists.

---

## Why this project

Language models can hallucinate confident-sounding but incorrect answers,
which is especially risky in a medical context. This project explores RAG as
a mitigation: instead of trusting a model's memorized knowledge, the system
retrieves real source passages first and forces the model to answer only
from that retrieved context — with an explicit fallback when retrieval finds
nothing relevant.

---

## Architecture

```
User (React frontend)
      │
      ▼
FastAPI backend  ──▶  Retrieval (FAISS + sentence-transformers)
      │                       │
      │                       ▼
      │              Diversity filtering
      │              (avoids duplicate chunks
      │               from the same source)
      │                       │
      │                       ▼
      │              Distance-threshold check
      │              (refuses to answer if no
      │               relevant match found)
      │                       │
      ▼                       ▼
Generation (flan-t5-base) ◀───┘
      │
      ▼
Grounded answer + cited sources
```

**Stack:**
- **Data:** MedQuAD — 16,407 real medical Q&A pairs from NIH sources
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`), 384-dimensional vectors
- **Vector search:** FAISS (`IndexFlatL2`)
- **Generation:** `flan-t5-base` (HuggingFace `transformers`)
- **Backend:** FastAPI
- **Frontend:** React

---

## Pipeline details

**1. Data preparation**
- Loaded 16,407 Q&A pairs; filtered 7 rows with degenerate (near-empty) answers
- Chunked long answers (max observed: 4,281 words) into ~300-word pieces
- Result: 20,612 retrieval-ready chunks

**2. Retrieval**
- Each chunk embedded into a 384-dim vector; indexed with FAISS
- **Diversity filtering:** retrieves a wider candidate pool (10) and keeps only
  the first chunk per unique source question, preventing near-duplicate
  chunks (common when one long answer is split into several pieces) from
  crowding out genuinely different relevant results
- **Confidence threshold:** if the closest retrieved match exceeds a distance
  of 1.0, the system returns "I don't have relevant information to answer
  that confidently" instead of generating an answer — verified against real
  distance measurements (~0.41 for genuinely relevant matches vs. ~1.46 for
  unrelated queries in testing)

**3. Generation**
- Retrieved chunks are inserted into a prompt template with the question
  placed *first*, so it survives truncation even if the model's 512-token
  context limit is exceeded by long retrieved passages
- Model generates an answer using only the provided context

---

## Evaluation

Two complementary metrics were used, because they measure different things
and neither alone tells the full story:

| Metric | Result | What it measures |
|---|---|---|
| Strict Accuracy@3 | 52% (52/100) | Did retrieval return the *exact* expected source chunk? |
| Relaxed (semantic similarity) Accuracy@3 | 80% (80/100) | Did retrieval return *content similar enough to be useful* (cosine similarity ≥ 0.6)? |

Both measured on 100 randomly sampled questions (fixed seed for
reproducibility) drawn from the dataset itself.

**Why the gap exists:** manual inspection of "failures" under the strict
metric showed many were not true misses — e.g., a question about *research*
on a cancer type retrieved information about *treatments* for the same
cancer. These are topically adjacent and often still useful, but the strict
exact-match metric penalizes them as failures. The relaxed metric better
reflects real-world usefulness.

---

## Bugs found and fixed during development

1. **Silent token truncation** — initial prompts exceeded the model's
   512-token limit, and with context placed before the question, the
   question itself risked being truncated away entirely — producing vague,
   context-less answers. Fixed by reordering the prompt (question first) and
   reducing retrieved chunk count.

2. **Hallucination on out-of-domain queries** — the system initially answered
   fluently even when retrieved context was completely irrelevant (e.g.,
   correctly answering "tallest mountain in the world" despite retrieving
   medical passages about a rare genetic disorder), because the model fell
   back on memorized training knowledge rather than the provided context.
   Fixed by adding a distance-based confidence threshold that refuses to
   generate an answer when no relevant chunk is found.

---

## Known limitations

- `flan-t5-base` is a small model; answers are often correct but terse.
  A larger model (or an API-based LLM) would likely improve fluency.
- The distance threshold (1.0) was set from a small number of manual
  examples; a larger, labeled test set would allow more rigorous tuning.
- Vector index is in-memory (FAISS); a production deployment would benefit
  from a persistent, scalable vector store (e.g., pgvector on PostgreSQL).

---

## Setup

**Backend:**
```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## Possible future work

- Swap in a larger/API-based LLM for improved answer fluency
- Migrate vector storage to pgvector for persistence and scale
- Expand evaluation to a larger, human-labeled test set
- Add answer-faithfulness scoring (e.g., via the RAGAS framework)
