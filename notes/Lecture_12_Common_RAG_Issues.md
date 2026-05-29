Here are **comprehensive lecture notes** for *Lecture 12: Common RAG Issues & Solutions*, based on the provided Jupyter notebook content. These notes cover every concept, debugging methodology, issue diagnosis, and fix.

---

## Lecture 12: Common RAG Issues & Solutions
**Course:** NLP with LangChain | **Platform:** Hope to Skill  
**Duration:** ~20 minutes | **Level:** Advanced  

---

## 1. The Big Picture

**Analogy: RAG Mechanic vs. Random Part Replacer**

| Mechanic Approach | RAG Equivalent |
|------------------|----------------|
| Listen to the engine | Read the RAGAS scores |
| Run diagnostics | Log retrieval + generation separately |
| Replace the bad part | Apply the targeted fix |
| Test drive | Re-run RAGAS to confirm |

> 🧠 **Key Insight:** Every RAG system has problems. The difference between a good engineer and a great one is knowing how to fix them **systematically**.

---

## 2. What You Will Learn

| # | Topic | What You'll Fix |
|---|-------|-----------------|
| 1 | The debugging mindset | How to think like a RAG mechanic |
| 2 | Poor retrieval quality | Wrong chunks coming back |
| 3 | Lost in the middle | LLM ignoring some retrieved docs |
| 4 | Hallucinations despite context | LLM making things up |
| 5 | Slow response times | Users waiting too long |
| 6 | Inconsistent results | Same question, different answers |
| 7 | Building a debugging toolkit | Your RAG maintenance kit |
| 8 | Hands-on: debug a broken RAG | Fix 3 real issues |

---

## 3. The Debugging Mindset

### The 4-Step Debugging Process

```
Step 1: ISOLATE       → Is it retrieval or generation?
Step 2: LOG           → What docs were retrieved? What prompt was sent?
Step 3: TEST          → Create test cases from failing queries
Step 4: FIX           → Change ONE variable at a time, then re-test
```

### The Golden Rule: Change ONE Variable at a Time

| Approach | What You Do | Result |
|----------|-------------|--------|
| **Bad (random)** | Change chunk_size, k, prompt, model all at once | No idea what helped |
| **Good (systematic)** | Change ONLY chunk_size, re-test, then try next | Know exactly what works |

> ⚠️ If you change 3 things and the score improves, you don't know which change actually helped.

### Quick Isolation Test

| Check | If Bad | Problem Is |
|-------|--------|-------------|
| Look at retrieved chunks — are the right ones there? | Wrong chunks retrieved | **Retrieval** |
| Right chunks retrieved, but answer is wrong? | LLM misused context | **Generation** |
| RAGAS `context_recall` is low? | Missing relevant info | **Retrieval** |
| RAGAS `faithfulness` is low? | LLM is hallucinating | **Generation** |

---

## 4. Build Base RAG System (Baseline)

### Components

```python
# Load and chunk
loader = TextLoader("data/nlp_article.txt")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Embeddings + Vector Store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name="nlp_course",
)

# Retriever (k=4)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Strong grounding prompt
good_prompt = ChatPromptTemplate.from_template(
    """You are a helpful teaching assistant for an NLP course.
Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't have enough information."

Context: {context}
Question: {question}
Answer:"""
)

# LLM with temperature=0
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

> This is the "healthy car" — we'll break things one at a time to learn diagnosis and fixes.

---

## 5. The Debug Helper (Diagnostic Scanner)

```python
def debug_rag(question, chain, ret, label="RAG"):
    """Run a question through RAG and show detailed debug info."""
    print(f"Question: {question}")
    
    # Step 1: Check retrieval
    docs = ret.invoke(question)
    for i, doc in enumerate(docs):
        print(f"  Chunk {i+1}: {doc.page_content[:100]}...")
    
    # Step 2: Get answer
    answer = chain.invoke(question)
    print(f"Answer: {answer}")
    
    return answer, docs
```

**What It Shows:**
| What It Shows | Why It Matters |
|---------------|----------------|
| Retrieved chunks | Are the right chunks being found? |
| Retrieval time | Is retrieval fast enough? |
| RAG answer | What did the LLM actually say? |
| Total time | Is the full pipeline fast enough? |

> 💡 **Pro tip:** In production, log this info for every query. When a user reports a bad answer, you can look up exactly what happened.

---

## 6. Issue 1: Poor Retrieval Quality

### Symptoms

| Symptom | What's Happening |
|---------|------------------|
| "I don't have enough information" | Retriever can't find the right chunks |
| Vague, incomplete answers | Retrieved chunks don't contain full info |
| Low RAGAS `context_recall` | Missing relevant information |

### Common Causes

| Cause | Effect |
|-------|--------|
| Chunk size too large | Important details buried in big chunks |
| Chunk size too small | Context split across multiple tiny chunks |
| `k` too low | Not retrieving enough chunks |
| Bad embedding model | Poor semantic matching |

### Experiment: How Chunk Size Affects Retrieval

| Chunk Size | Effect |
|------------|--------|
| 200 chars (too small) | More chunks, but each is a tiny fragment — context is split |
| 500 chars (good balance) | Each chunk has enough context to be meaningful |
| 1000 chars (large) | Fewer chunks, but may include unrelated info in same chunk |

### The Fix

| Setting | Recommendation |
|---------|----------------|
| `chunk_size` | Start with 500, test 300-1000 range |
| `chunk_overlap` | Use 10-20% of chunk_size (e.g., 50-100 for size=500) |
| `k` | Start with 4, increase to 7-10 if recall is low |

> 📌 **Key lesson:** There's no magic chunk size. Test different values and measure with RAGAS.

---

## 7. Issue 2: Lost in the Middle

### The Problem

The LLM **ignores** some of the retrieved documents, especially those in the middle of the context list. LLMs pay more attention to the beginning and end of long contexts.

### Symptoms

| Symptom | RAGAS Metric |
|---------|--------------|
| Answer uses only first retrieved chunk | Low `context_recall` |
| Answer misses key info from middle chunks | Low `faithfulness` |

### The Fix

| Fix | How It Helps |
|-----|--------------|
| Reduce `k` | Fewer chunks = less middle content to ignore |
| Reorder chunks | Put most important chunks first or last |
| Reranking | Use a reranker to put best chunks at top/bottom |

> 📌 **Key lesson:** LLMs have limited attention — don't overwhelm them with too many chunks.

---

## 8. Issue 3: Hallucinations Despite Context

### The Problem

The retriever finds the right chunks, but the LLM **makes up facts** that aren't in the context.

### Symptoms

| Symptom | RAGAS Metric |
|---------|--------------|
| Answer contains facts not in retrieved chunks | Low `faithfulness` |
| Answer mixes real info with made-up details | Low `faithfulness` |
| Answer says specific dates/numbers not in context | Low `faithfulness` |

### The #1 Cause: Weak Prompts

| Prompt | Risk |
|--------|------|
| "Answer the question using the context" | **High** — LLM may add its own knowledge |
| "Answer ONLY from the context. If not found, say 'I don't know'" | **Low** — LLM is constrained |

### Example: Weak vs Strong Prompt

**Weak Prompt (invites hallucination):**
```
Answer the question using the context below.
Context: {context}
Question: {question}
Answer:
```

**Strong Prompt (prevents hallucination):**
```
You are a helpful teaching assistant for an NLP course.
Answer the question based ONLY on the following context.
Do NOT add any information that is not explicitly stated in the context.
If the answer is not in the context, say "I don't have enough information."

Context: {context}
Question: {question}
Answer:
```

### The Anti-Hallucination Checklist

| Fix | What to Do |
|-----|-------------|
| Prompt | Add "ONLY use the context" and "say I don't know if not found" |
| Temperature | Set `temperature=0` (no randomness for factual answers) |
| System message | Reinforce grounding in the system prompt |
| Model choice | Some models are more faithful than others |

> ⚠️ **Warning:** Faithfulness is the #1 metric to get right. A RAG system that makes things up is worse than no RAG system at all.

### Temperature Comparison

| Temperature | Behavior | Use For |
|-------------|----------|---------|
| 0 | Same answer every time (deterministic) | Factual RAG, production systems |
| 0.7 | Slightly different each time (creative) | Creative writing, brainstorming |

> 📌 **Key lesson:** For factual RAG, always use `temperature=0`. Users expect the same question to give the same answer.

---

## 9. Issue 4: Slow Response Times

### The Problem

Users wait 5-10 seconds for answers. In production, anything over 3 seconds feels slow.

### Common Causes

| Cause | Why It's Slow |
|-------|---------------|
| Large `k` value | More chunks = bigger prompt = slower LLM |
| No streaming | User stares at a blank screen |
| Re-computing embeddings | Same query computed again and again |

### Fix 1: Use Streaming

Streaming doesn't make total time faster, but makes **user experience** much better.

| Method | User Experience |
|--------|-----------------|
| Regular | Blank screen for 3-5 seconds, then full answer appears |
| Streaming | Words start appearing in less than 1 second |

```python
# Streaming example
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)  # Words appear immediately
```

### Other Speed Fixes

| Fix | How It Helps | Example |
|-----|--------------|---------|
| Lower `k` | Less context = faster LLM response | `k=4` instead of `k=10` |
| Cache queries | Same question returns instant cached answer | Store results in a dict |
| Smaller model | Faster generation, slightly lower quality | `gpt-4o-mini` instead of `gpt-4o` |

> 📌 **Key lesson:** Always use streaming in user-facing applications. It's the single biggest UX improvement you can make.

---

## 10. Issue 5: Inconsistent Results

### The Problem

A user asks the same question twice and gets different answers. This destroys user trust.

### Causes & Fixes

| Cause | Fix |
|-------|-----|
| High temperature | Set `temperature=0` |
| Using "latest" model version | Pin exact model version |
| Borderline similarity scores | Add a minimum score threshold |
| Different question phrasing | Normalize questions before querying |

### Fix: Question Normalization

```python
def normalize_question(question):
    """Clean up a question for more consistent retrieval."""
    question = question.lower()                 # Remove case differences
    question = " ".join(question.split())       # Remove extra spaces
    question = question.strip()                 # Strip leading/trailing spaces
    return question
```

**Example:**

| Original | Normalized |
|----------|------------|
| `"What is BERT?"` | `"what is bert?"` |
| `"  what is BERT?  "` | `"what is bert?"` |
| `"WHAT IS BERT?"` | `"what is bert?"` |
| `"what   is   bert?"` | `"what is bert?"` |

> 📌 **Key lesson:** All variations normalize to the same string, so they get the same cache hit and same retrieval results.

---

## 11. Building Your Debugging Toolkit

### Essential Logging (Log These for Every Query)

| What to Log | Why |
|-------------|-----|
| Retrieved document IDs + scores | Know what the retriever found |
| Full prompt sent to LLM | See exactly what the LLM received |
| LLM response + latency | Track quality and speed |
| User question (normalized) | Track common queries |

### The Test Suite Approach

1. Create **20+ test questions** with expected answers
2. Run **RAGAS after every change**
3. Track **metric improvements** over time
4. **Document** what worked and what didn't

### The Change Log (Example)

| Date | Change Made | Metric Before | Metric After | Keep? |
|------|-------------|---------------|--------------|-------|
| Day 1 | chunk_size 500 → 300 | recall: 0.65 | recall: 0.72 | **Yes** |
| Day 2 | k=4 → k=8 | precision: 0.80 | precision: 0.60 | **No (revert)** |
| Day 3 | Added stronger prompt | faith: 0.75 | faith: 0.90 | **Yes** |

> 📌 **Key lesson:** Document everything. Your future self will thank you.

### Complete RAG Logger with Retry Logic

```python
def ask_and_log(question, chain, ret, log_list, max_retries=3):
    """Run a RAG query and log everything with retry logic."""
    normalized_q = normalize_question(question)
    
    for attempt in range(1, max_retries + 1):
        try:
            start = time.time()
            docs = ret.invoke(normalized_q)
            retrieval_time = time.time() - start
            
            start = time.time()
            answer = chain.invoke(question)
            total_time = time.time() - start
            
            log_entry = {
                "question": question,
                "normalized": normalized_q,
                "num_chunks": len(docs),
                "chunks_preview": [doc.page_content[:80] for doc in docs],
                "answer": answer,
                "retrieval_time": round(retrieval_time, 3),
                "total_time": round(total_time, 3),
            }
            log_list.append(log_entry)
            return answer
        except Exception as e:
            if attempt < max_retries:
                wait_time = attempt * 5
                print(f"[RETRY] Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                return f"[ERROR] Query failed after {max_retries} retries"
```

> 📌 **Key lesson:** Logging is not optional in production RAG. Without logs, you're debugging blind.

---

## 12. Hands-On: Debug a Broken RAG System

### The 3 Hidden Issues

| Issue # | Hint | What's Wrong |
|---------|------|--------------|
| 1 | Look at chunk size | `chunk_size=50` (tiny fragments) |
| 2 | Look at the prompt | Weak prompt (no grounding instruction) |
| 3 | Look at temperature | `temperature=1.0` (randomness) |

### Broken System Configuration

| Setting | Broken | Why It's Wrong |
|---------|--------|----------------|
| `chunk_size` | 50 | Too small — creates fragments, not meaningful text |
| `chunk_overlap` | 0 | No overlap — key info gets split |
| `k` | 3 | Too few chunks for complete answer |
| Prompt | Weak | No instruction to stay grounded in context |
| Temperature | 1.0 | Adds randomness and hallucination |

### Fixed System Configuration

| Setting | Fixed | Why It Works |
|---------|-------|--------------|
| `chunk_size` | 500 | Meaningful chunks with enough context |
| `chunk_overlap` | 50 | Smooth transitions between chunks |
| `k` | 4 | Enough chunks for complete answers |
| Prompt | Strong | "ONLY use the context" + "say I don't know" |
| Temperature | 0 | Deterministic, factual answers |

### RAGAS Comparison Results (Example)

| Metric | Broken | Fixed | Change |
|--------|--------|-------|--------|
| `faithfulness` | 0.0655 | 1.0000 | **+0.9345** |
| `answer_relevancy` | 0.8831 | 0.9225 | +0.0394 |
| `context_precision` | 0.0000 | 1.0000 | **+1.0000** |
| `context_recall` | 0.3333 | 1.0000 | **+0.6667** |

> 📌 **The data-driven optimization cycle:** Build → Evaluate → Fix → Re-evaluate → Ship

---

## 13. Quick Reference: RAG Debugging Cheat Sheet

### The 5 Most Common Issues

| # | Issue | Symptom | Metric | Fix |
|---|-------|---------|--------|-----|
| 1 | Poor retrieval | "I don't know" answers | Low `context_recall` | Adjust `chunk_size`, increase `k` |
| 2 | Lost in middle | LLM ignores some docs | Low `faithfulness` | Reduce `k`, reorder chunks |
| 3 | Hallucinations | Made-up facts | Low `faithfulness` | Strong prompt, `temp=0` |
| 4 | Slow responses | Users waiting >3s | N/A (timing) | Streaming, caching, lower `k` |
| 5 | Inconsistent | Different answers | Varies | `temp=0`, normalize questions |

### The Debugging Process (Code Template)

```python
# Step 1: Isolate the problem
docs = retriever.invoke(question)   # Check retrieval
answer = chain.invoke(question)     # Check generation

# Step 2: Log everything
print(f"Chunks: {len(docs)}")
for doc in docs:
    print(doc.page_content[:100])

# Step 3: Run RAGAS
result = evaluate(dataset=dataset, metrics=[faithfulness, ...])

# Step 4: Apply ONE fix, re-test
# Change chunk_size OR k OR prompt — not all at once!
```

### Production Checklist

- [ ] `temperature=0` for factual RAG
- [ ] Strong grounding prompt with "ONLY use context"
- [ ] `chunk_size` tested (300-1000 range)
- [ ] `k` value tested (3-7 range)
- [ ] Streaming enabled for user-facing apps
- [ ] Query logging enabled
- [ ] Test suite of 20+ questions with RAGAS
- [ ] Cache for repeated queries
- [ ] Question normalization

---

## 14. Key Takeaways

1. ✅ **Systematic debugging:** Always isolate (retrieval vs generation), log, test, then fix
2. ✅ **Change ONE variable at a time** — otherwise you won't know what helped
3. ✅ Most failures have **known causes and known fixes** — use the cheat sheet
4. ✅ **Logging + evaluation are essential** — you can't fix what you can't see
5. ✅ **Streaming** is the biggest UX win — always use it in user-facing apps
6. ✅ **Caching + normalization** save time and money in production

### The Complete RAG Optimization Flow

```
Build RAG (L10) → Evaluate with RAGAS (L11) → Debug & Fix (L12)
      ↑                                              |
      |______________________________________________|
              (measure, fix, re-measure, repeat)
```

---

## 15. Mini Challenges

### Challenge 1: Find the Best Chunk Size
Test chunk sizes: 300, 500, 700, 1000. Run RAGAS for each. Which gets the highest overall scores?

### Challenge 2: Build a Better Logger
Extend `ask_and_log()` to also log:
- The full prompt sent to the LLM
- Whether the answer contains "I don't have enough information"
- A warning if `total_time > 3` seconds

### Challenge 3: The Prompt Engineering Challenge
Write 3 different prompts and test them with RAGAS. Which gives the highest faithfulness score? Try:
- Different system roles
- Different grounding instructions
- Adding "think step by step"

> 💡 **Hint:** The best prompts are usually short, specific, and include clear constraints like "ONLY use the context."

---

## 16. PEP 8 Style Rules Used

| Category | Rule | Example |
|----------|------|---------|
| **Naming** | Variables/functions: `snake_case` | `debug_rag()`, `query_log` |
| | Constants: `UPPER_CASE` | `OPENAI_API_KEY` |
| **Imports** | One import per line | `from ragas import evaluate` |
| | Group: stdlib → third-party → local | `os, time` then `ragas` then `langchain` |
| **Whitespace** | Spaces around operators | `score >= 0.8` |
| | Space after commas | `evaluate(dataset=eval_data, metrics=[...])` |
| | Two blank lines before functions | See `debug_rag()` |
| **Best Practices** | f-strings for formatting | `f"Score: {score:.4f}"` |
| | `enumerate()` for loops | `for i, doc in enumerate(docs)` |
| | List comprehensions | `[doc.page_content[:80] for doc in docs]` |
| | `repr()` for debugging | `repr(original)` to show whitespace |
| | Docstrings for functions | Triple-quoted descriptions |
| | `time.time()` for timing | `start = time.time()` |

---

## Appendix: Required Imports Summary

```python
# Core RAG
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Evaluation
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Utilities
import os
import time
```

---

**End of Notes** — Hope to Skill: *Building the future, one skill at a time.*