# RAG Assignment: report notes

Numbers below come from `rag_assignment.ipynb`.

**Setup:** FIDE Laws of Chess (25 pages), 121 chunks of about 100 words (20-word overlap). Bi-encoder `all-MiniLM-L6-v2` with FAISS inner product, cross-encoder `ms-marco-MiniLM-L-6-v2`, LLM `gemini-3.1-flash-lite-preview` at temperature 0 with one shared prompt.

## 1. Architecture comparison (notes)
- Bi-encoder: query and chunk are each encoded alone into one vector. Score = cosine of the two vectors. Chunk vectors can be precomputed and indexed.
- Cross-encoder: query and chunk are concatenated into one input, so self-attention runs across all query and chunk tokens together. It outputs one relevance score. Nothing can be precomputed.
- Consequence: the bi-encoder compresses a whole chunk into one vector before it sees the query. The cross-encoder can match specific query words to specific chunk words (for example "always" against "cannot win... draw").

## 2. Rank shifts and answers

| # | Query | Gold chunk | Bi rank | CE rank | In top-3 before -> after |
|---|-------|-----------|---------|---------|--------------------------|
| 1 | Extra minutes when arbiter rejects a draw claim | 63 | 4 | 1 | No -> Yes |
| 2 | Opponent's time runs out, do I always win? (exception) | 65, 66 | 10 | 1 | No -> Yes |
| 3 | How long can a player be absent before forfeit? (synonym trap) | 32, 33 | 5 | 9 | No -> No |
| 4 | Allowed to withdraw a draw claim? (negation) | 59 | 1 | 1 | Yes -> Yes |
| 5 | Prohibited to write moves in advance? (negation) | 47 | 1 | 1 | Yes -> Yes |
| 6 | Third illegal move (control) | 45 | 1 | 1 | Yes -> Yes |
| 7 | Mobile phones | 65, 72 | 1 | 1 | Yes -> Yes |

Answers:
- **Query 2 (the clear win):** Answer 1 was "Not found in the provided context." The gold chunk (66, Article 12.3 with the "unless the opponent cannot win... draw" clause) sat at bi-encoder rank #10. Answer 2 correctly said that if the opponent cannot win by any series of legal moves, the result is a draw.
- **Query 1:** the retrieval improved (#4 to #1), but Answer 1 was already correct ("two extra minutes") because chunk 62 held a partly overlapping clause. The retrieval gain did not change the answer.
- **Keyword trap (query 3):** the cross-encoder did not help. The gold clause (Article 6.7, default time) moved from #5 to #9 and neither answer used it. Both answers described the adjourned-game rule instead. Likely cause: the query says "absent" and "forfeited", while the rule says "arrives late" and "default time", and the MS-MARCO cross-encoder preferred the chunk with "late" and "lose".
- **Negation queries (4 and 5):** the bi-encoder already ranked the right chunk #1, so I could not show a negation failure on this document. Be honest about that.
- Answer differences on queries 4 to 7 are wording only, except that Answer 1 for phones added the "unless the opponent cannot win... draw" clause and Answer 2 dropped it. That clause comes from chunk 66, which fell out of the top 3 after re-ranking, so here re-ranking slightly narrowed the answer.

## 3. Latency (mean over 7 queries x 10 repeats, CPU/MPS, 10 candidates)

| Stage | Time |
|-------|------|
| Pipeline 1: bi-encoder retrieval (embed query + FAISS top-10) | 5.0 ms |
| Re-ranking step (10 cross-encoder pairs) | 27.4 ms |
| Pipeline 2 retrieval total (bi + CE) | 32.4 ms (6.5x baseline) |
| LLM generation (one uncached call, same for both pipelines) | about 1.1 s |
| Cross-encoder over all 121 chunks (illustration) | 398 ms |

The LLM call dominates end-to-end time, so the extra 27 ms of re-ranking is small in practice. The last row grows linearly with the corpus size.

## 4. Scaling question (notes)
- A cross-encoder needs a full transformer pass for every (query, chunk) pair, and nothing can be cached because the score depends on the query. Cost is O(N) transformer passes per query.
- Bi-encoder chunk vectors are computed once offline, and a query costs one encoding plus a fast (approximate) nearest-neighbour search that is sublinear in N.
- 121 chunks cost 398 ms. A million chunks would cost minutes per query. So: cheap recall stage first (top-k), expensive precision stage on only k candidates.
- Limit of this design: if the gold chunk is not in the top-k, the re-ranker cannot recover it.

## Honest caveats
- Only 7 queries, one document, one chunking setting. Two queries show a clear promotion; this is not a statistically meaningful benchmark.
- The gold chunks were located by phrase search after reading the rulebook.
- Gemini free-tier quota (20 requests per day per model) forced answer caching (`llm_cache.json`).
