# 🧠 Complete RAG System — Senior Gen AI Developer Interview Notes

> **How to use this doc:** Read it top to bottom once. Then use the headers as flash-card prompts. If an interviewer says "walk me through a production RAG system," you can narrate this entire document.

---

## Table of Contents

1. [What Is RAG & Why It Exists](#1-what-is-rag--why-it-exists)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Step 1 — Ingestion & Normalization](#3-step-1--ingestion--normalization)
4. [Steps 2–4 — Hybrid Retrieval, ANN, Reranking](#4-steps-24--hybrid-retrieval-ann--reranking)
5. [Step 4b — Source Confidence Scoring](#5-step-4b--source-confidence-scoring)
6. [Steps 5–7 — Constrained Generation, Citations & Hallucination Detection](#6-steps-57--constrained-generation-citations--hallucination-detection)
7. [Step 8 — Continuous Evaluation](#7-step-8--continuous-evaluation)
8. [Step 9 — Caching & Conversation Memory](#8-step-9--caching--conversation-memory)
9. [Step 10 — Observability & Tracing](#9-step-10--observability--tracing)
10. [Performance Benchmarks](#10-performance-benchmarks)
11. [Advanced Topics & Trade-offs](#11-advanced-topics--trade-offs)
12. [Common Interview Questions & Model Answers](#12-common-interview-questions--model-answers)
13. [Terminology Cheat Sheet](#13-terminology-cheat-sheet)

---

## 1. What Is RAG & Why It Exists

**RAG = Retrieval-Augmented Generation**

A technique that grounds LLM responses in *real, verifiable documents* rather than relying on baked-in parametric knowledge.

### The Core Problem RAG Solves

| Problem | Without RAG | With RAG |
|---|---|---|
| Knowledge cutoff | Model doesn't know post-training facts | Retrieves live/updated docs |
| Hallucination | Model confidently fabricates | Constrained to retrieved context |
| Source attribution | No citations possible | Every claim linked to a source |
| Domain specificity | Generic answers | Answers grounded in your corpus |
| Cost | Fine-tuning is expensive | Just update the doc store |

### Mental Model

```
User Query → Retrieve Relevant Docs → Inject into Prompt → LLM Generates Grounded Response
```

RAG sits between **pure prompting** (fast, no grounding) and **fine-tuning** (expensive, static). It's the pragmatic production choice for most enterprise use cases.

---

## 2. System Architecture Overview

```
[User Query]
     │
     ├── Cache Hit? ──YES──→ Return Cached Response (< 1ms)
     │
     NO
     │
     ▼
[Step 1]  Ingestion & Normalization
     │
     ▼
[Steps 2-4] Hybrid Retrieval → ANN → Reranking → Confidence Scoring
     │
     ▼
[Steps 5-7] Constrained Generation → Citations → Hallucination Check
     │
     ▼
[Steps 8-10] Eval + Caching + Observability (run in parallel)
     │
     ▼
[Final Output: Response + Citations + Confidence + Trace]
```

### Two Pipelines Inside RAG

1. **Offline / Indexing Pipeline** — runs when documents change. Steps 1 → chunk → embed → store in vector DB.
2. **Online / Query Pipeline** — runs on every user query. Steps 2–10.

> **Interview tip:** Always distinguish these two pipelines. Confusing them is a red flag.

---

## 3. Step 1 — Ingestion & Normalization

> *"Garbage in, garbage out. The quality of your RAG system is bounded by your ingestion pipeline."*

### What Happens

Raw documents enter as JSON, plain text, PDFs, or API responses. This step transforms them into a clean, uniform, chunked, versioned knowledge base.

### Sub-Steps in Detail

#### 3.1 Deduplication

**Why:** Duplicate chunks cause the same evidence to be double-counted, inflating confidence scores.

**How:**
- Compute **SHA-256 content hash** of each document.
- Before storing, check if hash already exists in the index.
- Exact duplicates → discard. Near-duplicates → more nuanced (see MinHash/SimHash below).

**Techniques:**
- **Exact dedup:** SHA-256 hash comparison — O(1) lookup.
- **Near-dedup:** MinHash + LSH (Locality Sensitive Hashing) — finds semantically similar docs even with minor edits.
- **Semantic dedup:** Embed docs and cluster; remove docs within cosine distance threshold (e.g., > 0.95 similarity).

#### 3.2 Format Standardization

Every source format → one **Unified Document Schema**:

```json
{
  "id": "uuid-v4",
  "content": "raw text",
  "source": "official_docs | research_paper | blog | forum",
  "url": "https://...",
  "created_at": "ISO-8601",
  "version": 3,
  "metadata": { ... }
}
```

**Why this matters in interviews:** Interviewers love asking "how do you handle heterogeneous sources?" This is your answer — normalize early, reason uniformly.

#### 3.3 Metadata Extraction

Extracted automatically per document:

| Metadata Field | Method | Use |
|---|---|---|
| Word count | `len(text.split())` | Chunk sizing decisions |
| Character count | `len(text)` | Token estimation |
| Entities | Regex / NER | Email, URLs, named entities for filtering |
| Freshness | `datetime.now() - created_at` | Freshness scoring (Step 4) |
| Language | langdetect / fastText | Multi-lingual routing |
| Source type | Rule-based classification | Trust scoring (Step 4) |

#### 3.4 Versioning

- Every document update increments a version number.
- Old versions are **retained** (not deleted) for audit trails and rollback.
- A "current version" pointer is maintained per document ID.
- Enables **temporal queries** — "What did we know about X on date Y?"

#### 3.5 Chunking — The Most Critical Decision

**Why chunk at all?** LLMs have token limits. Embedding models (e.g., `text-embedding-3-small`) have a max of 8191 tokens. You want each chunk to be semantically focused — one idea per chunk.

**Chunking Strategies:**

| Strategy | How It Works | Best For |
|---|---|---|
| **Fixed-size** | Split every N tokens | Simple baseline |
| **Sentence-based** | Split at sentence boundaries | Conversational content |
| **Paragraph-based** | Split at `\n\n` | Articles, blogs |
| **Semantic chunking** | Embed sentences; split where cosine distance drops | Dense technical docs |
| **Recursive** | Try paragraph → sentence → word in order | General purpose (LangChain default) |
| **Document-aware** | Respect headings/sections | Structured docs (Markdown, HTML) |

**Target chunk size:** 50–200 words (roughly 100–300 tokens).

**Chunk Overlap:** Add 10–20% overlap between adjacent chunks to avoid cutting context at boundaries.

```
Chunk 1: [words 1-150]
Chunk 2: [words 130-280]  ← 20-word overlap
Chunk 3: [words 260-410]
```

**Parent-Child Chunking (Advanced):**
- Store small chunks (child) for retrieval precision.
- Return the parent (larger context) to the LLM for generation.
- Best of both: precise retrieval + coherent context.

> **Interview trap:** "What chunk size do you use?" Correct answer: *"It depends on the domain and embedding model. I'd run ablations."* Never say a fixed number without justification.

---

## 4. Steps 2–4 — Hybrid Retrieval, ANN & Reranking

> *"Retrieval is the most impactful lever in a RAG system. A 10% improvement in retrieval quality beats a 10% improvement in generation quality every time."*

### Step 2 — Hybrid Retrieval

Run **three complementary retrieval methods in parallel**, then merge.

#### 2a. BM25 — Keyword Search

**Full name:** Best Match 25 (probabilistic IR model)

**How it works:**
```
BM25(q, d) = Σ IDF(qᵢ) × [f(qᵢ, d) × (k1 + 1)] / [f(qᵢ, d) + k1 × (1 - b + b × |d|/avgdl)]
```

- `f(qᵢ, d)` = term frequency of query term in document
- `IDF` = inverse document frequency (rare terms score higher)
- `k1` (1.2–2.0), `b` (0.75) = tunable parameters
- `avgdl` = average document length (normalizes for doc length)

**Strengths:** Exact keyword matching, fast, no GPU needed, great for named entities, model names, acronyms.

**Weaknesses:** Synonyms not handled ("car" ≠ "automobile"), no semantic understanding.

**Implementation:** Elasticsearch, OpenSearch, BM25Okapi in `rank_bm25` Python library.

#### 2b. Semantic (Dense) Retrieval

**How it works:**
1. Embed query → query vector `q ∈ ℝ^d`
2. All chunk vectors pre-stored in vector DB
3. Find chunks where `cosine_similarity(q, chunk_vector)` is highest

**Embedding models to know:**

| Model | Dimensions | Notes |
|---|---|---|
| `text-embedding-3-small` (OpenAI) | 1536 | Good balance, cheap |
| `text-embedding-3-large` (OpenAI) | 3072 | Best quality |
| `bge-large-en` (BAAI) | 1024 | Open-source, strong |
| `e5-mistral-7b` | 4096 | SOTA, expensive |
| `nomic-embed-text` | 768 | Fast, open-source |

**Strengths:** Captures semantic meaning, synonyms, paraphrases.

**Weaknesses:** Can miss exact keywords, embeddings are static after training, expensive at scale.

**Vector DBs:** Pinecone, Weaviate, Qdrant, Chroma, pgvector (Postgres), FAISS (library).

#### 2c. ANN — Approximate Nearest Neighbor

**Why ANN instead of exact NN?** Exact nearest neighbor search over millions of vectors is O(N) — too slow. ANN trades tiny accuracy loss for massive speed gain.

**ANN Algorithms:**

| Algorithm | How | Speed | Accuracy |
|---|---|---|---|
| **HNSW** (Hierarchical Navigable Small World) | Graph-based navigation | Very fast | Very high |
| **IVF** (Inverted File Index) | Cluster vectors, search only nearest cluster | Fast | High |
| **PQ** (Product Quantization) | Compress vectors, approx distance | Very fast | Medium |
| **IVF-PQ** | Combination | Fastest | Medium |
| **ScaNN** (Google) | Anisotropic quantization | Fastest | High |

**HNSW is the de facto standard** in most vector DBs (Weaviate, Qdrant use it).

**Key HNSW parameters:**
- `M` (connections per node): higher → better recall, more memory
- `ef_construction`: higher → better index quality, slower build
- `ef_search`: higher → better recall at query time, slower

#### Merging Hybrid Results — RRF

**Reciprocal Rank Fusion (RRF):**

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```
Where `k=60` (constant), `rank_i(d)` is rank of doc `d` in retrieval method `i`.

- Simple, robust, doesn't require score normalization.
- Works better than weighted sum in practice.

**Alternative: Weighted Linear Combination:**
```
hybrid_score = α × semantic_score + (1-α) × bm25_score
```
Typical `α = 0.7` (favor semantic).

### Step 3 — Reranking

**Problem:** The merged top-50 results from hybrid retrieval aren't perfectly ordered.

**Solution:** Apply a **cross-encoder reranker** on the top-50 to get a precise top-5.

**Bi-encoder vs Cross-encoder:**

| | Bi-encoder (retrieval) | Cross-encoder (reranking) |
|---|---|---|
| Input | Query + Doc separately | Query + Doc concatenated |
| Speed | Fast (pre-compute doc vectors) | Slow (must run per pair) |
| Accuracy | Good | Excellent |
| Use case | First-stage retrieval | Second-stage reranking |

**Cross-encoder workflow:**
```
[CLS] query [SEP] document [SEP] → BERT → [CLS] representation → sigmoid → relevance score
```

**Popular rerankers:**
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, open-source)
- `Cohere Rerank API` (production, plug-and-play)
- `BGE-Reranker-Large` (SOTA open-source)

**Why only rerank top-50 not all docs?**
Rerankers are O(n) — running on millions of docs is prohibitive. Two-stage retrieval (fast retrieval → precise reranking) is the standard pattern.

### Query Expansion & Transformation (Advanced)

Before retrieval, improve the query itself:

1. **HyDE (Hypothetical Document Embeddings):** Use LLM to generate a *hypothetical answer*, embed that, retrieve against it. Better semantic signal than raw query.
   ```
   User: "How does HNSW work?"
   HyDE: Generate fake answer → embed → retrieve
   ```

2. **Query Decomposition:** Break complex queries into sub-queries, retrieve for each, merge.
   ```
   "Compare HNSW vs IVF performance on 10M vectors"
   → Sub-query 1: "HNSW performance characteristics"
   → Sub-query 2: "IVF performance characteristics"
   ```

3. **Step-back Prompting:** Abstract the query to a higher level before retrieval.

4. **Multi-query:** Generate N paraphrases of the query, retrieve for each, deduplicate results.

---

## 5. Step 4b — Source Confidence Scoring

After retrieval, score each chunk on three dimensions before passing to generation.

### Formula

```
Final Confidence = (Relevance × 0.5) + (Trust × 0.3) + (Freshness × 0.2)
```

### Relevance Score (0–1)

Derived from retrieval:
- BM25 normalized score
- Cosine similarity from semantic search
- Reranker score (most reliable after reranking)

### Trust Score (by source type)

| Source Type | Trust Score | Rationale |
|---|---|---|
| Official documentation | 0.95 | Authoritative, maintained |
| Peer-reviewed research paper | 0.85 | Validated, rigorous |
| Internal company docs | 0.80 | Domain-specific authority |
| Technical blog (reputable) | 0.60 | Informative but opinionated |
| Forum (Stack Overflow etc.) | 0.50 | Community wisdom, variable quality |
| Unknown/unclassified | 0.40 | Default fallback |

### Freshness Score

| Age | Score | Rationale |
|---|---|---|
| < 30 days | 1.0 | Current |
| 30–90 days | 0.8 | Mostly current |
| 90–365 days | 0.4 | May be stale |
| > 1 year | 0.2 | Likely outdated |

> **Note:** For timeless content (math theorems, historical facts), freshness should be **ignored** or down-weighted. This is a nuance interviewers love.

### Confidence Threshold

**Filter rule:** Drop any chunk with final confidence < 0.3

**Why 0.3?** Low enough to not miss relevant content, high enough to filter noise. This is a hyperparameter — tune it using your evaluation metrics from Step 8.

### Why This Matters

Without confidence scoring:
- Stale blog posts compete equally with official docs
- Forum speculation gets equal weight as research papers
- LLM can't signal "I'm not sure" vs "I'm confident"

With confidence scoring:
- Each citation carries a confidence level
- Users can calibrate trust in the response
- System can fall back gracefully when all scores are low

---

## 6. Steps 5–7 — Constrained Generation, Citations & Hallucination Detection

### Step 5 — Constrained Generation

**The Prompt Architecture:**

```
[System Prompt]
You are a precise assistant. Answer ONLY using the provided context.
Do NOT use external knowledge. Do NOT make assumptions.
If the context doesn't contain the answer, say "Insufficient evidence."
Always cite your sources using [Source N] notation.

[Context Block]
Source 1 (official_docs, confidence: 0.91): "..."
Source 2 (research_paper, confidence: 0.78): "..."
Source 3 (blog, confidence: 0.62): "..."

[User Query]
{user_question}
```

**Why "constrained" generation?**
- Standard LLMs will fill gaps with parametric knowledge — you can't control what they "remember."
- Explicit constraints in the system prompt reduce (not eliminate) this.
- The hallucination detection step (Step 7) is your safety net.

**Prompt Engineering Techniques:**

| Technique | Effect |
|---|---|
| Explicit "only use context" instruction | Reduces parametric knowledge leakage |
| Negative examples ("do NOT...") | More effective than positive constraints alone |
| Chain-of-thought ("think step by step") | Improves reasoning quality |
| XML/structured output format | Easier to parse citations programmatically |
| Temperature = 0 | Deterministic output, easier to test |

**Advanced: Structured Output**

Force the LLM to output JSON:
```json
{
  "answer": "...",
  "citations": [{"source_id": 1, "claim": "..."}],
  "confidence": 0.87,
  "uncertainty_flags": []
}
```
Use OpenAI function calling, Instructor library, or `response_format: { type: "json_object" }`.

### Step 6 — Citations

Every factual claim in the response is linked to a source chunk.

**Citation format options:**

1. **Inline:** "The model achieves 94% accuracy [Source 1, Page 5]."
2. **Footnote:** "The model achieves 94% accuracy.¹"
3. **Structured JSON:** Machine-readable for downstream processing.

**Why citations matter (business case):**
- Legal/compliance: "Where did this answer come from?"
- User trust: Verifiable answers increase adoption
- Debugging: Trace wrong answers to their source
- Audit: Required in regulated industries (healthcare, finance, legal)

**Implementation challenge:** LLMs don't always cite accurately even when instructed. Post-process the response to verify each cited source actually contains the attributed claim (string matching or semantic similarity check).

### Step 7 — Hallucination Detection

**What is hallucination in RAG?**
The model makes claims not grounded in the retrieved context — either from parametric memory or confabulation.

**Detection Signals:**

| Signal | Example | Risk |
|---|---|---|
| Hedge words | "probably", "might", "I think", "typically" | Medium — uncertainty without basis |
| Missing citation | Claim made without [Source N] | High |
| Citation mismatch | [Source 1] cited but claim not in Source 1 | Very High |
| Numeric fabrication | Statistics not in context | Very High |
| Temporal confusion | "Currently" when source is 2 years old | Medium |

**Hallucination Risk Score Calculation:**

```python
risk_score = 0.0

# Check hedge words
hedge_words = ["probably", "might", "I think", "usually", "typically"]
if any(word in response.lower() for word in hedge_words):
    risk_score += 0.3

# Check missing citations
claims = extract_claims(response)
uncited_claims = [c for c in claims if not has_citation(c)]
risk_score += len(uncited_claims) * 0.2

# Cap at 1.0
risk_score = min(1.0, risk_score)
```

**Threshold:** If `risk_score > 0.5` → use fallback response.

**Fallback Response:**
```
"Based on the provided context, I cannot find sufficient evidence to answer this question confidently. 
The available sources discuss [related topics], but do not directly address your query. 
Please consult [source types] for authoritative information."
```

**Advanced Hallucination Mitigation:**

1. **NLI-based checking:** Use a Natural Language Inference model to check if each claim is *entailed* by its cited source. (`premise: source text`, `hypothesis: claim` → entailment/neutral/contradiction).
2. **Self-consistency:** Generate 3–5 responses at temperature > 0. If they disagree, flag as high uncertainty.
3. **Faithfulness scoring:** Use tools like RAGAS `faithfulness` metric or TruLens.
4. **Atomic claim decomposition:** Break response into atomic claims, verify each against context.

---

## 7. Step 8 — Continuous Evaluation

> *"You can't improve what you don't measure. RAG eval is notoriously hard — most teams skip it and pay the price."*

### Retrieval Metrics

**Precision@K:** Of the top-K retrieved chunks, what fraction are actually relevant?
```
Precision@K = |relevant ∩ retrieved_top_K| / K
```

**Recall@K:** Of all relevant chunks in the corpus, what fraction did we retrieve in top-K?
```
Recall@K = |relevant ∩ retrieved_top_K| / |all_relevant|
```

**F1@K:** Harmonic mean of Precision and Recall.
```
F1@K = 2 × (Precision@K × Recall@K) / (Precision@K + Recall@K)
```

**MRR (Mean Reciprocal Rank):** Rewards finding the right answer higher in the ranking.
```
MRR = (1/|Q|) × Σ 1/rank_i
```

**NDCG (Normalized Discounted Cumulative Gain):** Accounts for graded relevance (not just binary).

### Generation Metrics

| Metric | What It Measures | Tool |
|---|---|---|
| **Faithfulness** | Is response grounded in context? | RAGAS |
| **Answer Relevance** | Does response answer the question? | RAGAS |
| **Context Precision** | Were the retrieved chunks relevant? | RAGAS |
| **Context Recall** | Did retrieval find necessary info? | RAGAS |
| **Hallucination Rate** | % responses with ungrounded claims | TruLens |
| **BLEU / ROUGE** | Lexical overlap with reference answer | Traditional NLP |
| **BERTScore** | Semantic similarity to reference | Hugging Face |
| **G-Eval** | LLM-as-judge evaluation | PromptFlow |

### RAGAS — The Gold Standard RAG Evaluation Framework

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
```

### Human Evaluation

Automated metrics have blind spots. Supplement with:
- **Relevance labeling:** Human annotators mark chunks as relevant/irrelevant for sampled queries.
- **A/B testing:** Deploy two RAG configs, measure user satisfaction / task completion.
- **Error taxonomy:** Categorize failures (wrong retrieval vs wrong generation vs wrong chunking).

### Target Benchmark Numbers (Production)

| Metric | Minimum Acceptable | Good | Excellent |
|---|---|---|---|
| Precision@5 | 0.60 | 0.75 | 0.90 |
| Recall@5 | 0.55 | 0.70 | 0.85 |
| F1@5 | 0.57 | 0.72 | 0.87 |
| Faithfulness | 0.70 | 0.85 | 0.95 |
| Answer Relevance | 0.75 | 0.85 | 0.92 |

---

## 8. Step 9 — Caching & Conversation Memory

### Query Cache

**Purpose:** Avoid recomputing expensive retrieval + generation for repeated queries.

**Implementation:**

```python
cache = {}  # In production: Redis with TTL

def query_with_cache(query: str, ttl_seconds: int = 3600):
    cache_key = hashlib.sha256(query.encode()).hexdigest()
    
    if cache_key in cache:
        cache["hits"] += 1
        return cache[cache_key]["result"]
    
    result = full_rag_pipeline(query)
    cache[cache_key] = {"result": result, "expires_at": time() + ttl_seconds}
    return result
```

**Cache Key Options:**
- Exact query hash (fast, misses paraphrases)
- Normalized query hash (lowercase, strip punctuation)
- Semantic cache: embed query, find cache hits where cosine similarity > 0.95 (GPTCache library)

**TTL Strategy:**
- Static content: 24h TTL
- Semi-dynamic content: 1h TTL
- Real-time data: No cache or very short TTL (< 5min)

**Performance Impact:**
- Cache miss: 1.5–6.5 seconds
- Cache hit: < 1ms
- Cost reduction: ~10x for frequently repeated queries

**Semantic Caching (Advanced):**
```
Incoming query → embed → find nearest cached query embedding
If cosine_similarity > 0.95: return cached result
Else: run full pipeline, cache result
```
Libraries: GPTCache, LangChain's `CacheBackedEmbeddings`.

### Conversation Memory

**Problem:** RAG is stateless by default. Multi-turn conversations lose context.

**Memory Types:**

| Type | Storage | Use Case |
|---|---|---|
| **Buffer memory** | Last N messages | Simple chatbots |
| **Summary memory** | Compressed summary of history | Long conversations |
| **Entity memory** | Extract and track named entities | CRM-style interactions |
| **Vector memory** | Embed past turns, retrieve relevant ones | Complex long-term memory |

**Implementation (Buffer Memory):**
```python
conversation_memory = deque(maxlen=10)  # Last 10 query-response pairs

def build_context_with_memory(current_query: str, retrieved_chunks: list):
    memory_context = "\n".join([
        f"Previous Q: {pair['query']}\nPrevious A: {pair['response']}"
        for pair in list(conversation_memory)[-3:]  # Last 3 turns
    ])
    return memory_context + "\n\n" + format_chunks(retrieved_chunks)
```

**Multi-turn Query Rewriting:**
Before retrieval, rewrite the current query to be self-contained:
```
History: "Tell me about HNSW" → "HNSW is a graph-based ANN algorithm..."
Current: "How does it compare to IVF?"

Rewritten: "How does HNSW compare to IVF for approximate nearest neighbor search?"
```
This dramatically improves retrieval quality in multi-turn conversations.

---

## 9. Step 10 — Observability & Tracing

> *"In production, you are blind without observability. A RAG system that you can't trace is a liability."*

### Distributed Tracing

Every query generates a **trace** — a tree of spans recording each operation.

**Trace Structure:**
```
[Query: "How does HNSW work?"] → Total: 2,473ms
├── [ingest_check] 2ms
├── [cache_lookup] 1ms (MISS)
├── [retrieve]
│   ├── [bm25_search] 45ms
│   ├── [semantic_search] 62ms
│   ├── [ann_search] 13ms
│   └── [rerank] 120ms
├── [confidence_scoring] 8ms
├── [generate_response] 2,200ms
│   ├── [prompt_build] 5ms
│   ├── [llm_call] 2,190ms
│   └── [hallucination_check] 5ms
└── [cache_write + eval_update + trace_store] 22ms
```

### Key Metrics to Track

**Latency:**
- P50 (median), P95, P99 latency per operation
- Don't use average — it hides tail latency problems

**Throughput:**
- Queries per second
- Tokens per second (generation)

**Quality:**
- Hallucination rate (rolling 7-day)
- Cache hit rate
- Retrieval precision (sampled)
- User thumbs up/down ratio

**Cost:**
- LLM tokens per query
- Embedding tokens per ingestion
- Vector DB query units

**Error Rates:**
- Retrieval timeout rate
- LLM API error rate
- Fallback response rate (high fallback rate = retrieval or chunking problem)

### Tooling

| Tool | Use Case |
|---|---|
| **LangSmith** | LangChain tracing & evaluation |
| **Arize Phoenix** | LLM observability, hallucination monitoring |
| **TruLens** | RAG evaluation & tracing |
| **Weights & Biases** | Experiment tracking |
| **Prometheus + Grafana** | Infrastructure metrics |
| **OpenTelemetry** | Vendor-neutral distributed tracing |
| **Datadog** | Full-stack observability |

### Alerting Rules

Set alerts for:
- P95 latency > 8 seconds
- Hallucination rate > 15%
- Cache hit rate drops > 20% (sudden new query patterns)
- LLM error rate > 2%
- Fallback response rate > 30% (retrieval breakdown)

---

## 10. Performance Benchmarks

| Step | Min | Typical | Max | Optimization |
|---|---|---|---|---|
| Step 1: Ingestion | 5ms | 20ms | 100ms | Async, batch processing |
| Steps 2-4: Retrieval | 30ms | 100ms | 500ms | ANN, caching embeddings |
| Step 5: Generation | 1,000ms | 2,500ms | 5,000ms | Streaming, smaller models |
| Steps 6-7: Safety | 5ms | 20ms | 50ms | Lightweight NLI models |
| Step 8: Evaluation | 1ms | 5ms | 20ms | Async, non-blocking |
| Step 9: Caching | <1ms | <1ms | 5ms | Redis in-memory |
| Step 10: Tracing | <1ms | <1ms | 5ms | Async writes |
| **TOTAL (first)** | **1.5s** | **3.0s** | **6.5s** | |
| **TOTAL (cached)** | **<1ms** | **<1ms** | **5ms** | |

### Optimization Strategies

1. **Streaming:** Stream LLM tokens to user immediately — perceived latency drops dramatically even if total latency is the same.
2. **Async retrieval:** Run BM25, semantic, and ANN in parallel (not sequential).
3. **Batch embedding:** Group ingestion jobs, embed in batches of 100+.
4. **Smaller models for safety checks:** Don't use GPT-4 for hallucination detection — a fine-tuned `distilbert` is 100x faster.
5. **Pre-compute embeddings:** Never re-embed unchanged documents.
6. **Quantized vectors:** INT8 quantization reduces vector DB memory by 4x with <2% accuracy loss.

---

## 11. Advanced Topics & Trade-offs

### RAG vs Fine-tuning vs Prompt Engineering

| Approach | When to Use | Cost | Freshness | Control |
|---|---|---|---|---|
| Prompt engineering | Simple tasks, LLM already knows | Low | Static | Low |
| RAG | Domain knowledge, live data, citations needed | Medium | Dynamic | High |
| Fine-tuning | Style/tone adaptation, task-specific behavior | High | Static | Medium |
| RAG + Fine-tuning | Best quality, both grounding and style | Highest | Dynamic | Highest |

### Chunking vs Embedding Model Alignment

Your chunk size should match what the embedding model was trained on. Most models were trained on ~256 tokens. Chunking at 512 tokens with a 256-token embedding model degrades quality.

### The "Lost in the Middle" Problem

Research shows LLMs perform worst on information in the **middle** of a long context. Mitigation:
- Keep context short (top-5 chunks, not top-20)
- Put most relevant chunks first and last
- Use reranking to put the best chunk first

### Handling Multi-hop Questions

Some questions require information from multiple documents:
```
"Who is the CEO of the company that acquired OpenAI's main competitor in 2023?"
```
Standard RAG fails here. Solutions:
1. **Iterative retrieval:** Retrieve → generate intermediate answer → retrieve again.
2. **Graph RAG:** Build a knowledge graph from documents, traverse it for multi-hop reasoning.
3. **IRCoT (Interleaved Retrieval and Chain-of-Thought):** Alternate between retrieval and reasoning steps.

### Security Considerations

1. **Prompt injection via documents:** Adversarial documents can contain instructions to the LLM. Sanitize retrieved content.
2. **Data exfiltration:** Ensure retrieved context doesn't leak across user tenants (multi-tenancy isolation in vector DB).
3. **PII in chunks:** Scrub PII before ingestion or implement access control at retrieval time.
4. **Adversarial queries:** Queries designed to extract training data or bypass constraints.

### Scaling Considerations

| Scale | Architecture |
|---|---|
| Prototype (<10K docs) | In-memory FAISS, SQLite for BM25 |
| Small prod (<1M docs) | Chroma/Qdrant + Elasticsearch |
| Medium prod (<100M docs) | Pinecone/Weaviate + Elasticsearch + Redis cache |
| Large prod (>100M docs) | Distributed vector DB, sharding, tiered storage |

---

## 12. Common Interview Questions & Model Answers

**Q: What's the difference between RAG and fine-tuning?**

A: RAG retrieves external knowledge at inference time — the model's weights don't change. Fine-tuning bakes knowledge into model weights during training. RAG is better for dynamic knowledge, citations, and cost efficiency. Fine-tuning is better for style consistency and when you want the model to behave differently, not just know different things. In practice, they're complementary.

---

**Q: How do you choose chunk size?**

A: It's an empirical decision, not a fixed answer. I'd start with 256 tokens (matching most embedding models' training distribution), then run evaluation with your actual queries. Measure retrieval precision and recall at different chunk sizes. Technical documentation often benefits from smaller chunks (100–150 tokens) for precision; conversational content from larger chunks (300–500 tokens) for coherence. Always include overlap.

---

**Q: How do you evaluate a RAG system?**

A: Two dimensions — retrieval and generation. For retrieval: Precision@K, Recall@K, MRR. For generation: faithfulness (is it grounded?), answer relevance, and hallucination rate. I'd use RAGAS as the framework. I'd also track business metrics: user satisfaction, task completion rate, and escalation rate (how often do users give up and call a human).

---

**Q: How do you handle hallucinations?**

A: Multiple layers. In prompt design: explicit constraints to only use context. In post-processing: detect hedge words, check citation coverage, NLI-based faithfulness checking. At the system level: confidence scoring to filter low-quality context, fallback responses when confidence is low, and monitoring hallucination rate in production to catch regressions.

---

**Q: What happens when the retrieved context is insufficient?**

A: The system should gracefully degrade — return a fallback response that acknowledges the limitation rather than hallucinating. I'd also implement a "low confidence" signal that the frontend can use to show a disclaimer. The root cause is usually either (a) the document wasn't indexed, (b) chunking broke the relevant passage, or (c) the retrieval query didn't match. Each requires a different fix.

---

**Q: How would you improve retrieval quality?**

A: In order of impact: (1) better chunking — most teams underinvest here; (2) query expansion (HyDE, multi-query); (3) better embedding model; (4) hybrid retrieval instead of pure semantic; (5) cross-encoder reranking; (6) metadata filtering to pre-narrow the search space. Measure retrieval recall after each change.

---

**Q: How do you handle multi-tenant RAG (different users have different document access)?**

A: Each document gets access control metadata at ingestion. At retrieval time, filter by `user_id` or `tenant_id` before semantic search. Vector DBs like Weaviate and Qdrant support metadata filters that are applied before ANN search (pre-filtering) or after (post-filtering). Pre-filtering is safer (no data leakage) but may reduce recall. Alternatively, maintain separate vector DB namespaces per tenant.

---

**Q: Walk me through what happens when a user asks a question.**

A: (1) Check cache — exact hash lookup. Hit → return cached response in <1ms. Miss → continue. (2) Optional ingestion check — if new docs need indexing, run Step 1. (3) Hybrid retrieval — BM25 and semantic search run in parallel, results merged via RRF. (4) Cross-encoder reranking of top-50 → get top-5. (5) Confidence scoring per chunk — filter anything < 0.3. (6) Build prompt with system constraints + context + query. (7) LLM generates response with citations. (8) Hallucination check — if risk > threshold, return fallback. (9) Store in cache. (10) Log trace, update eval metrics. Return response + citations + confidence + latency to user.

---

## 13. Terminology Cheat Sheet

| Term | Definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation — grounding LLM with external documents |
| **Chunk** | A segment of a document, typically 100-300 tokens |
| **Embedding** | Dense vector representation of text in semantic space |
| **Vector DB** | Database optimized for similarity search over embeddings |
| **BM25** | Probabilistic keyword retrieval algorithm (TF-IDF variant) |
| **ANN** | Approximate Nearest Neighbor — fast similarity search |
| **HNSW** | Hierarchical Navigable Small World — best ANN algorithm |
| **Bi-encoder** | Embeds query and document separately (fast, used for retrieval) |
| **Cross-encoder** | Processes query+doc together (slow, used for reranking) |
| **Reranking** | Second-stage precision scoring of retrieved results |
| **RRF** | Reciprocal Rank Fusion — merging results from multiple retrievers |
| **HyDE** | Hypothetical Document Embeddings — LLM-augmented query expansion |
| **Hallucination** | LLM claim not grounded in context |
| **NLI** | Natural Language Inference — checks if claim is entailed by source |
| **RAGAS** | RAG Assessment framework — standard eval metrics |
| **Faithfulness** | Metric: are all claims in response grounded in context? |
| **TTL** | Time-to-live — cache expiration period |
| **Trace** | Full log of spans/operations for a single query |
| **P95 latency** | 95th percentile latency — 95% of requests are faster than this |
| **Semantic cache** | Cache using embedding similarity instead of exact key match |
| **Parent-child chunking** | Retrieve small chunks, return their larger parent for generation |
| **IRCoT** | Interleaved Retrieval + Chain-of-Thought for multi-hop reasoning |
| **Graph RAG** | Uses a knowledge graph instead of flat vector search |
| **Confidence score** | Weighted combination of relevance, trust, and freshness |

---

*Built for senior Gen AI engineering interviews. Every concept here maps to a real production decision.*
