# RAG Assignment: Baseline Retrieval vs Cross-Encoder Re-Ranking

Compares two retrieval pipelines over the same chunked PDF (FIDE Laws of Chess, `doc/laws_of_chess.pdf`):

1. **Baseline:** bi-encoder top-10 -> top-3 chunks -> LLM (Answer 1)
2. **Re-ranking:** the same top-10 -> cross-encoder re-sort -> top-3 chunks -> LLM (Answer 2)

Both use the same prompt, so any difference in the answers comes from which chunks were retrieved.

## Files

| File | Contents |
|------|----------|
| `rag_assignment.ipynb` | The full pipeline with outputs: chunking, both pipelines, failure-mode queries, rank shifts, latency |
| `report.tex`, `report.pdf` | The written report (source and compiled PDF) |
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

Nine queries (selected from 40+ candidates to expose failures, so these are examples and not a benchmark):

- **Promoted into the top-3:** the gold chunk entered the top-3 for 6 of 9 queries, for example rank 10 to 1 ("do I always win?"), rank 9 to 1 (negation: "claim a draw after touching a piece") and rank 9 to 2 (synonym: "referee" against "arbiter").
- **Answers:** for 3 of those queries the baseline answered "Not found in the provided context" and the re-ranked pipeline answered correctly. The baseline failed by omission, not by hallucination.
- **Where it did not help:** on the synonym query "absent... forfeited" the cross-encoder moved the right chunk from rank 5 to 9. On the scoresheet query the chunk was promoted but both answers were still "Not found".
- **Latency:** baseline retrieval about 8 ms, re-ranking adds about 28 ms, the LLM call about 0.7 s. Running the cross-encoder over all 121 chunks took about 380 ms.

See `report.pdf` for the full tables and analysis.

## Limitations

Seven queries on one document is a demonstration, not a benchmark. The gold chunks were found by searching the rulebook text for the relevant clause.
