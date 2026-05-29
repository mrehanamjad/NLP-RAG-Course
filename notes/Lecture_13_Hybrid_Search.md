Here are **comprehensive lecture notes** for *Lecture 13: Dense and Sparse Vectors in Search (Hybrid Search)*, based on the provided Jupyter notebook content. These notes cover every concept, important point, code pattern, and best practice.

---

## Lecture 13: Dense and Sparse Vectors in Search (Hybrid Search)
**Course:** NLP with LangChain | **Platform:** Hope to Skill  
**Duration:** ~15 minutes | **Level:** Intermediate  

---

## 1. Overview

**The Problem:** Dense vector search (embeddings) is powerful for understanding meaning, but it has limitations — especially with exact matches like product codes, chemical names, and IDs.

**The Solution:** Combine **dense** (semantic) and **sparse** (keyword) search into **hybrid search** — the best of both worlds.

> 🔥 **Key Insight:** Production RAG systems use hybrid search to handle all query types.

---

## 2. What You Will Learn

| # | Topic |
|---|-------|
| 1 | Why dense search alone is not enough |
| 2 | What sparse vectors are and when they excel |
| 3 | How to create a dense-only vector store in Qdrant |
| 4 | How to create a hybrid vector store (dense + sparse) in Qdrant |
| 5 | When to use each approach |

---

## 3. Why Dense Search Is Not Enough

### The Problem with Dense Vectors

Dense vectors (embeddings) capture **semantic meaning** but struggle with **exact matches**.

### Manual Example 1: Dense Search Captures Meaning ✅

```python
query1 = "heart attack symptoms"
query2 = "signs of myocardial infarction"
query3 = "cardiac arrest warning signs"
# Dense search understands: These queries mean similar things.
# Result: All three find similar documents about heart problems.
```

### Manual Example 2: Dense Search Fails on Exact Matches ❌

```python
query = "3-fluorophenmetrazine"

# Document 1: "The substance 3-fluorophenmetrazine is a stimulant"
# Document 2: "Fluorinated stimulant compounds include phenmetrazine derivatives"
# Document 3: "Synthetic cathinones and related stimulants"

# Dense search problem: All three documents have similar embeddings
# (stimulants, chemicals). Dense search might return Document 2 or 3 instead
# of Document 1, even though only Document 1 contains the exact chemical name.
```

### When Dense Search Struggles

| Query Type | Example | Dense Search Performance |
|------------|---------|--------------------------|
| Exact names | "flubromazepam" | Poor — returns similar chemicals |
| Product codes | "Model XR-500" | Poor — returns similar products |
| CAS numbers | "439-14-5" | Poor — treats as random numbers |
| IDs | "Error E-404" | Poor — focuses on "error" concept |
| Natural language | "What causes headaches?" | **Excellent** — understands meaning |

---

## 4. The Solution: Sparse Vectors

Sparse vectors (also called **keyword** or **BM25 search**) work like traditional search engines:

- Find documents containing **exact word matches**
- Count how many times keywords appear
- Score **rare words higher** (e.g., "3-fluorophenmetrazine" is rare, "the" is common)

### Manual Example 3: Sparse Search Excels on Exact Matches

```python
query = "3-fluorophenmetrazine"

# Sparse search looks for EXACT word match
# Document 1: "The substance 3-fluorophenmetrazine is a stimulant" → MATCH! High score
# Document 2: "Fluorinated stimulant compounds include phenmetrazine" → Partial match, lower score
# Document 3: "Synthetic cathinones and related stimulants" → No match

# Sparse search returns Document 1 first because it contains the exact keyword.
```

### Dense vs Sparse: Feature Comparison

| Search Type | Strengths | Weaknesses |
|-------------|-----------|------------|
| **Dense** (embeddings) | Understands meaning, handles synonyms | Misses exact matches |
| **Sparse** (keywords) | Finds exact words, codes, names | Doesn't understand meaning |
| **Hybrid** (dense + sparse) | Gets **both** meaning AND exact matches | None — best of both! |

> 📌 **Key lesson:** Production RAG systems use hybrid search to handle all query types.

---

## 5. Building the Knowledge Base

For this lecture, we use the **WHO Substances Under Surveillance PDF** — a document containing:
- **Exact chemical names** (e.g., "3-fluorophenmetrazine") — needs sparse search
- **Health descriptions** (e.g., "opioid-like effects") — needs dense search

### Load and Split the Document

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("data/who_substances_surveillance.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(documents)
```

---

## 6. Create Dense-Only Vector Store

First, let's create a traditional vector store using only dense vectors.

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

dense_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name="who_dense_only",
    force_recreate=True,
)

# Test with exact chemical name (dense search struggles)
query = "3-fluorophenmetrazine"
results = dense_store.similarity_search(query, k=3)
# May not find the exact match first
```

---

## 7. Create Hybrid Vector Store (Dense + Sparse)

Now let's create a collection that uses **both** dense and sparse vectors. Qdrant supports hybrid search natively.

### Create Collection with Both Vector Types

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

client.create_collection(
    collection_name="who_hybrid",
    vectors_config={
        "dense": models.VectorParams(
            size=384,  # HuggingFace embedding size
            distance=models.Distance.COSINE,
        )
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=False),
        )
    },
)
```

### Create Sparse Vector Helper (BM25-style)

```python
def create_sparse_vector(text):
    """Create a simple sparse vector from text using word frequency."""
    words = text.lower().split()
    word_counts = {}
    for word in words:
        word = word.strip('.,!?;:"\'')
        if len(word) > 2:  # Ignore very short words
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Convert to sparse vector format with unique indices
    index_values = {}
    for word, count in word_counts.items():
        idx = hash(word) % 100000
        index_values[idx] = index_values.get(idx, 0.0) + float(count)
    
    indices = sorted(index_values.keys())
    values = [index_values[idx] for idx in indices]
    
    return models.SparseVector(indices=indices, values=values)
```

### Index Documents with Both Vector Types

```python
points = []
for i, chunk in enumerate(chunks):
    # Get dense embedding
    dense_vector = embeddings.embed_query(chunk.page_content)
    
    # Create sparse vector
    sparse_vector = create_sparse_vector(chunk.page_content)
    
    point = models.PointStruct(
        id=i,
        vector={
            "dense": dense_vector,
            "sparse": sparse_vector,
        },
        payload={"text": chunk.page_content, "source": chunk.metadata.get("source", "")},
    )
    points.append(point)

client.upsert(collection_name="who_hybrid", points=points)
```

### Test Hybrid Search

```python
# Query with exact chemical name
query = "3-fluorophenmetrazine"
query_dense = embeddings.embed_query(query)
query_sparse = create_sparse_vector(query)

# Dense search only
dense_results = client.query_points(
    collection_name="who_hybrid",
    query=query_dense,
    using="dense",
    limit=3,
)

# Sparse search only
sparse_results = client.query_points(
    collection_name="who_hybrid",
    query=query_sparse,
    using="sparse",
    limit=3,
)
```

---

## 8. LangChain Hybrid Search (EnsembleRetriever) — RECOMMENDED

**EnsembleRetriever** is LangChain's built-in way to combine multiple retrievers using **Reciprocal Rank Fusion (RRF)**.

### How RRF Works

Instead of combining different similarity scores, RRF combines the **rank positions**:

```
RRF_score(doc) = weight1/(k + rank_in_retriever1) + weight2/(k + rank_in_retriever2)
```

Where **k=60** is a constant that prevents top documents from dominating too much.

### Why Use EnsembleRetriever?

| Approach | Implementation | Use Case |
|----------|---------------|----------|
| Qdrant native hybrid | Requires Qdrant-specific code | When you control the vector DB |
| **LangChain EnsembleRetriever** | Works with ANY retrievers | **Production standard** |

> 📌 **Key insight:** EnsembleRetriever is framework-agnostic — combine ANY retrievers (BM25, Qdrant, Pinecone, etc.)

### Building EnsembleRetriever

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Retriever 1: BM25 (Sparse - Keyword Search)
bm25_retriever = BM25Retriever.from_documents(documents=chunks)
bm25_retriever.k = 3

# Retriever 2: Qdrant (Dense - Semantic Search)
qdrant_retriever = dense_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Combine with EnsembleRetriever (Hybrid)
hybrid_retriever = EnsembleRetriever(
    retrievers=[qdrant_retriever, bm25_retriever],
    weights=[0.5, 0.5],  # Equal weight: 50% dense, 50% sparse
)

# Use in RAG pipeline
results = hybrid_retriever.invoke("3-fluorophenmetrazine")
```

### Comparing All Three Retrievers

```python
# Test 1: Exact chemical name (sparse should excel)
query = "3-fluorophenmetrazine"

dense_results = qdrant_retriever.invoke(query)   # May struggle
sparse_results = bm25_retriever.invoke(query)    # Should find exact match
hybrid_results = hybrid_retriever.invoke(query)  # Best of both

# Test 2: Semantic query (dense should excel)
query2 = "Which substances cause opioid-like effects?"

dense_results2 = qdrant_retriever.invoke(query2)   # Understands meaning
sparse_results2 = bm25_retriever.invoke(query2)    # Keyword matching only
hybrid_results2 = hybrid_retriever.invoke(query2)  # Combines both
```

---

## 9. Key Takeaways

### When to Use Each Approach

| Approach | Best For | Example |
|----------|----------|---------|
| **Dense-only** | Natural language questions, semantic search | Customer support chatbot |
| **Sparse-only** | Exact codes, product catalog, legal citations | Product manual search |
| **Hybrid** | Technical docs, medical databases, e-commerce | **All production RAG systems** |

### Implementation Patterns

**Pattern 1: Dense-only (Simple)**
```python
dense_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="my_collection",
)
retriever = dense_store.as_retriever(search_kwargs={"k": 3})
```

**Pattern 2: LangChain Hybrid (RECOMMENDED)**
```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

dense_retriever = dense_store.as_retriever(search_kwargs={"k": 3})

hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.5, 0.5],
)

results = hybrid_retriever.invoke("your query")
```

**Pattern 3: Qdrant Native Hybrid (Advanced)**
```python
client.create_collection(
    collection_name="hybrid_collection",
    vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams(...)},
)
```

### Production Recommendations

| Recommendation | Why |
|---------------|-----|
| **Start with EnsembleRetriever** | Works with any LangChain-compatible retriever |
| **Use equal weights [0.5, 0.5]** | Adjust based on your specific use case |
| **Monitor performance** | Track which retriever contributes most to good results |
| **Consider Qdrant native** | If you need maximum performance and control |

> 📌 **Final recommendation:** For most production RAG systems, use LangChain's **EnsembleRetriever** for flexibility and ease of use.

---

## 10. Quick Reference: Hybrid Search Cheat Sheet

### Dense vs Sparse vs Hybrid

| Feature | Dense | Sparse | Hybrid |
|---------|-------|--------|--------|
| Best for | Meaning, synonyms | Exact matches | **All query types** |
| Example queries | "heart attack symptoms" | "3-fluorophenmetrazine" | Both! |
| Vector size | Fixed (384-1536 dims) | Variable (non-zero words) | Both |
| Production use | Good | Good | **Best!** |

### Two Approaches to Hybrid Search

| Approach | Pros | Cons |
|----------|------|------|
| **Qdrant Native** | Fast, single DB query | Vendor-specific code |
| **LangChain EnsembleRetriever** | Framework-agnostic, flexible | Two separate queries |

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 3-5 | Number of documents to retrieve |
| `weights` | [0.5, 0.5] | Balance between dense and sparse |
| `RRF constant k` | 60 | Prevents top documents from dominating |

---

## 11. PEP 8 Style Rules Used

| Category | Rule | Example |
|----------|------|---------|
| **Naming** | Variables/functions: `snake_case` | `dense_store`, `create_sparse_vector()` |
| | Constants: `UPPER_CASE` | `QDRANT_URL`, `QDRANT_API_KEY` |
| | Classes: `PascalCase` | `QdrantClient`, `VectorParams` |
| **Whitespace** | Spaces around operators | `size = 384` |
| | Space after commas | `[1, 2, 3]` |
| **Best Practices** | f-strings for formatting | `f"Result: {score}"` |
| | Descriptive variable names | `dense_vector` not `dv` |
| | Comments explain WHY, not WHAT | |
| | One import per line | |

---

## Appendix: Required Imports Summary

```python
# Document loading & splitting
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings & vector stores
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Qdrant client (for native hybrid)
from qdrant_client import QdrantClient, models

# LangChain retrievers (for EnsembleRetriever)
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Utilities
import os
```

---

**End of Notes** — Hope to Skill: *Building the future, one skill at a time.*