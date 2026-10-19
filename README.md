# RAG Assignment: Baseline Retrieval vs Cross-Encoder Re-Ranking

Compares two retrieval pipelines over the same chunked PDF (FIDE Laws of Chess, `doc/laws_of_chess.pdf`):

1. **Baseline:** bi-encoder top-10 -> top-3 chunks -> LLM (Answer 1)
2. **Re-ranking:** the same top-10 -> cross-encoder re-sort -> top-3 chunks -> LLM (Answer 2)

Both use the same prompt, so any difference in the answers comes from which chunks were retrieved.

## Files

| File | Contents |
|------|----------|
| `rag_assignment.ipynb` | The full pipeline with outputs: chunking, both pipelines, failure-mode queries, rank shifts, latency |
| `report.md` | Results tables and notes for the four report questions |
| `doc/laws_of_chess.pdf` | Source document |
| `llm_cache.json` | Cached Gemini answers, so the notebook can be rerun without spending API quota |
| `requirements.txt` | Python dependencies |

## Setup

| Part | Choice |
|------|--------|
| Chunking | about 100 words, 20-word overlap, 121 chunks |
| Bi-encoder | `sentence-transformers/all-MiniLM-L6-v2`, FAISS inner product on normalised vectors |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `gemini-3.1-flash-lite-preview`, temperature 0 |

## Running it

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key      # read from the environment, never stored in the repo
jupyter notebook rag_assignment.ipynb
```

With `llm_cache.json` present, the stored answers are reused and no Gemini calls are made. Without it, a full run makes about 15 requests, which is close to the free-tier limit of 20 per day per model.

## Results in brief

- **Rank shifts:** for the query on extra minutes after a rejected draw claim, the right chunk moved from rank 4 to 1. For "if my opponent's time runs out, do I always win?", the chunk with the exception moved from rank 10 to 1.
- **Answers:** in the second case the baseline answered "Not found in the provided context", while the re-ranked pipeline gave the correct draw exception. In the first case both answers were correct.
- **Where it did not help:** on a synonym-heavy query ("absent... forfeited" against "late... default time") the cross-encoder moved the right chunk from 5 to 9. On the negation queries the baseline already ranked the right chunk first.
- **Latency:** baseline retrieval about 5 ms, re-ranking adds about 27 ms, the LLM call about 1.1 s. Running the cross-encoder over all 121 chunks took about 400 ms.

See `report.md` for the full tables.

## Limitations

Seven queries on one document is a demonstration, not a benchmark. The gold chunks were found by searching the rulebook text for the relevant clause.
