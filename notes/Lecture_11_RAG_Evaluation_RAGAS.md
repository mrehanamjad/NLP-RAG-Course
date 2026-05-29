Here are **comprehensive lecture notes** for *Lecture 11: RAG Evaluation with RAGAS*, based on the provided Jupyter notebook content. These notes cover every concept, important point, code pattern, and best practice.

---

## Lecture 11: RAG Evaluation with RAGAS
**Course:** NLP with LangChain | **Platform:** Hope to Skill  
**Duration:** ~20 minutes | **Level:** Advanced  

---

## 1. The Big Picture

**The Problem:** In Lecture 10 you built a working RAG system. But how do you know if it's actually good? Is it answering correctly? Is it making things up?

**Analogy:** Think of RAGAS as a **report card** for your RAG system. Just like a school report card grades you in Math, Science, and English, RAGAS grades your RAG in:
- **Faithfulness** (Is it truthful?)
- **Relevancy** (Does it answer the question?)
- **Precision** (Are retrieved chunks useful?)
- **Recall** (Did we find everything relevant?)

Each grade tells you exactly what needs improvement.

> 📌 **Key quote:** *"You cannot improve what you do not measure." — Peter Drucker*

---

## 2. What You Will Learn

| # | Topic | Real-World Analogy |
|---|-------|--------------------|
| 1 | Why evaluation matters | Why students need exams |
| 2 | Retrieval metrics (precision & recall) | Did you find the right books? |
| 3 | Generation metrics (faithfulness) | Did you copy the answer honestly? |
| 4 | Generation metrics (relevancy & context) | Did you actually answer the question? |
| 5 | What is RAGAS? | The automatic grading machine |
| 6 | Creating test datasets | Writing the exam questions |
| 7 | Hands-on: run a full evaluation | Grade your RAG system! |
| 8 | Common issues & fixes | How to improve each grade |

---

## 3. Why Evaluation Matters

RAG systems can fail in **two completely different ways**:

| Failure Mode | What Went Wrong | Example |
|--------------|-----------------|---------|
| **Bad Retrieval** | Retrieved the wrong chunks | Asked about "BERT" but got chunks about "cooking recipes" |
| **Bad Generation** | Right chunks, but LLM hallucinated | Chunks mention 2017, but answer says 2020 |

### Without Evaluation: Guessing Game

```
User: "The answer was wrong!"
You:  "Was it the retrieval or the generation?"
You:  "...I have no idea."  😕
```

### With Evaluation: Data-Driven Fixing

```
RAGAS Report:
  Faithfulness:      0.92  (great!)
  Context Recall:    0.45  (problem!)

You: "The retrieval is missing relevant chunks.
      I need to increase k or improve chunking."
```

> 📌 **Key insight:** Evaluation turns guessing into knowing. You can pinpoint exactly where your RAG system is weak and fix it.

---

## 4. Retrieval Metrics: Precision & Recall

These metrics measure **how well your retriever finds the right documents**.

### Context Precision: "Of the chunks I retrieved, how many were actually useful?"

| Retrieved 5 Chunks | Relevant? |
|--------------------|-----------|
| Chunk about transformers | Yes ✓ |
| Chunk about cooking | No ✗ (noise!) |
| Chunk about BERT | Yes ✓ |
| Chunk about weather | No ✗ (noise!) |
| Chunk about GPT | Yes ✓ |

**Precision = 3 relevant / 5 retrieved = 0.60** (too much noise)

| Score | Meaning |
|-------|---------|
| High precision (>0.8) | Almost every retrieved chunk is useful |
| Low precision (<0.6) | Too many irrelevant chunks cluttering the context |

### Context Recall: "Of all the relevant chunks that exist, how many did I find?"

| All Relevant Chunks in DB | Retrieved? |
|---------------------------|------------|
| Chunk about transformers (Chapter 3) | Yes ✓ |
| Chunk about attention mechanism | No ✗ (missed!) |
| Chunk about BERT (Chapter 3) | Yes ✓ |
| Chunk about self-attention | No ✗ (missed!) |

**Recall = 2 found / 4 total relevant = 0.50** (missing important info)

| Score | Meaning |
|-------|---------|
| High recall (>0.8) | Found almost everything relevant |
| Low recall (<0.6) | Missing important information |

### Simple Way to Remember

| Metric | Slogan |
|--------|--------|
| **Precision** | Low noise (no junk in results) |
| **Recall** | Nothing missed (found everything important) |

---

## 5. Generation Metric: Faithfulness (The Most Critical!)

**Faithfulness measures:** Is the answer grounded in the retrieved context?

This is the **single most important metric** for RAG systems.

| Score | Meaning | Example |
|-------|---------|---------|
| 1.0 | Every claim in the answer is supported by the context | Perfect! |
| 0.8 | Most claims are supported, minor additions | Good |
| 0.5 | Half the answer is made up | Concerning |
| 0.0 | The answer is completely hallucinated | Terrible! |

### Example

**Context:** *"The transformer was introduced by Vaswani et al. in 2017."*

| Answer | Faithfulness | Why |
|--------|--------------|-----|
| "The transformer was introduced by Vaswani et al. in 2017." | **1.0** | Directly from context |
| "The transformer was introduced in 2017 and revolutionized NLP." | **0.5** | "revolutionized NLP" not in context |
| "The transformer was created by Google Brain in 2015." | **0.0** | Wrong year, wrong team |

> 🎯 **Target for Production:** Faithfulness > 0.8 is the minimum. If faithfulness is below 0.8, your RAG system is "lying" to users. **Fix this FIRST** before worrying about other metrics.

---

## 6. Generation Metrics: Relevancy & Context

### Answer Relevancy: "Does the answer actually address the question?"

| Question | Answer | Relevancy |
|----------|--------|-----------|
| "What is BERT?" | "BERT is a language model by Google." | **High** |
| "What is BERT?" | "Transformers use attention." | **Low** (didn't answer the question) |
| "What is BERT?" | "I like pizza." | **0.0** (completely off-topic) |

### All 4 Metrics Summary

| Metric | Measures | Target | What to Fix if Low |
|--------|----------|--------|-------------------|
| **Faithfulness** | Is answer grounded in context? | > 0.8 | Stronger prompt, temperature=0 |
| **Answer Relevancy** | Does answer address the question? | > 0.7 | Improve prompt specificity |
| **Context Precision** | Are retrieved chunks useful? | > 0.7 | Add filters, use reranking |
| **Context Recall** | Did we find all relevant chunks? | > 0.7 | Increase k, improve chunking |

### The Chain of Dependency

```
Good retrieval (precision + recall) feeds good generation (faithfulness + relevancy).
If retrieval is bad, generation can't be good — garbage in, garbage out.
```

---

## 7. What Is RAGAS?

**RAGAS = Retrieval Augmented Generation Assessment**

It's an open-source framework that **automatically evaluates your RAG system** using LLMs — no human labelers needed!

### How RAGAS Works

```
You provide:                          RAGAS returns:
                                      
  question          ----+
  retrieved context ----+--->  Faithfulness:       0.92
  RAG answer        ----+       Answer Relevancy:   0.88
  ground truth      ----+       Context Precision:  0.75
                                Context Recall:     0.80
```

### What Makes RAGAS Special

| Feature | Details |
|---------|---------|
| **Automated** | Uses LLMs to evaluate — no manual scoring |
| **Comprehensive** | Scores both retrieval AND generation |
| **Per-question** | Shows scores for each question individually |
| **Actionable** | Low score on a specific metric tells you exactly what to fix |
| **Industry standard** | Used by top AI teams at major companies |

### The 4 Inputs RAGAS Needs

| Input | What It Is | Where It Comes From |
|-------|-----------|---------------------|
| `question` | The user's question | Your test dataset |
| `contexts` | The retrieved chunks | Your retriever |
| `answer` | The RAG-generated answer | Your RAG chain |
| `ground_truth` | The expected correct answer | You write this manually |

---

## 8. Running RAGAS Evaluation (Step-by-Step)

### Step 1: Build the RAG System (Review from Lecture 10)

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Knowledge base
loader = TextLoader("data/nlp_article.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = QdrantVectorStore.from_documents(
    documents=chunks, embedding=embeddings,
    url=QDRANT_URL, api_key=QDRANT_API_KEY,
    collection_name="nlp_course"
)

# RAG Chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_template(
    """Answer the question based ONLY on the following context.
    If the answer is not in the context, say "I don't have enough information."
    Context: {context}
    Question: {question}
    Answer:"""
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt | llm | StrOutputParser()
)
```

### Step 2: Create the Test Dataset

```python
test_questions = [
    "What is Natural Language Processing?",
    "When was the transformer architecture introduced?",
    # ... more questions
]

ground_truths = [
    "NLP is a branch of AI that focuses on interaction between computers and humans...",
    "The transformer architecture was introduced in 2017 by Vaswani et al....",
    # ... more ground truths
]
```

### Step 3: Run RAG on All Test Questions

```python
answers = []
contexts = []

for question in test_questions:
    answer = rag_chain.invoke(question)
    answers.append(answer)
    
    retrieved_docs = retriever.invoke(question)
    context_texts = [doc.page_content for doc in retrieved_docs]
    contexts.append(context_texts)
```

### Step 4: Run RAGAS Evaluation

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

eval_data = {
    "question": test_questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
}

eval_dataset = Dataset.from_dict(eval_data)

result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
)

# View results
df = result.to_pandas()
print(df.mean())  # Overall average scores
```

---

## 9. Understanding Your RAGAS Report Card

### Sample Output

```
============================================================
RAG SYSTEM REPORT CARD
============================================================
  faithfulness              0.9000  (GREAT)
  answer_relevancy          0.8307  (GREAT)
  context_precision         1.0000  (GREAT)
  context_recall            0.8600  (GREAT)
============================================================
```

### Score Interpretation

| Score Range | Status |
|-------------|--------|
| > 0.8 | **GREAT** — This part is working well |
| 0.6 - 0.8 | **OK** — Room for improvement |
| < 0.6 | **NEEDS WORK** — This is your weakest link; fix this first |

### Per-Question Analysis

Look for **individual questions with low scores** — they reveal your RAG's specific weak spots.

---

## 10. Common Issues & Fixes

| Problem | Metric | Target | Fix |
|---------|--------|--------|-----|
| RAG is hallucinating | Faithfulness < 0.8 | > 0.8 | Stronger prompt, temperature=0, add "only use context" |
| Missing relevant info | Context Recall < 0.7 | > 0.7 | Increase k, smaller chunk_size |
| Too much noise in context | Context Precision < 0.6 | > 0.7 | Metadata filters, reranking, better embeddings |
| Answer doesn't match question | Answer Relevancy < 0.7 | > 0.7 | More specific prompt |
| Everything is low | All metrics < 0.7 | > 0.7 | Check embeddings match, verify document content |

### Priority Order for Fixing

```
1. Faithfulness    (fix first — your RAG is lying!)
2. Context Recall  (fix second — you're missing information)
3. Context Precision (fix third — too much noise)
4. Answer Relevancy (fix last — usually improves with better context)
```

---

## 11. Automatic Diagnosis Tool

```python
fixes = {
    "faithfulness": [
        "Add 'Answer ONLY from the context' to your prompt",
        "Set temperature=0 (no creativity for factual RAG)",
        "Use a stronger LLM (e.g., gpt-4o instead of gpt-4o-mini)",
    ],
    "answer_relevancy": [
        "Make your prompt more specific about answering the question",
        "Add 'Be concise and directly answer the question' to prompt",
        "Check if the context contains enough information",
    ],
    "context_precision": [
        "Add metadata filters to narrow retrieval",
        "Use a reranker to push relevant chunks to the top",
        "Try a better embedding model (e.g., OpenAI embeddings)",
    ],
    "context_recall": [
        "Increase k (retrieve more chunks, e.g., k=10)",
        "Use smaller chunk_size (e.g., 300 instead of 500)",
        "Increase chunk_overlap to avoid splitting key info",
    ],
}

# Find weakest metric
weakest = min(metric_scores, key=metric_scores.get)
print(f"Weakest metric: {weakest}")
print(f"Recommended fixes: {fixes[weakest]}")
```

---

## 12. Creating Test Datasets: Manual vs Synthetic

| Method | How | Best For |
|--------|-----|----------|
| **Manual** | Write 20-50 questions + answers | Most reliable, full control |
| **Synthetic** | RAGAS generates questions from documents | Fast, covers more ground |

### Best Practice

| Stage | Approach | Number of Questions |
|-------|----------|---------------------|
| Start | Manual | 10-20 |
| Expand | Add synthetic | 50-100 |
| Validate | Spot-check synthetic | 10-20 |
| Production | Mix of both | 100+ |

> 📌 **For this course:** Manual questions are sufficient. Synthetic generation is useful for large document collections.

---

## 13. Quick Reference: RAGAS Evaluation Cheat Sheet

### The 4 RAGAS Metrics

| Metric | Measures | Target | Fix if Low |
|--------|----------|--------|-------------|
| **Faithfulness** | Is answer grounded in context? | > 0.8 | Stronger prompt, temp=0 |
| **Answer Relevancy** | Does answer address question? | > 0.7 | More specific prompt |
| **Context Precision** | Are retrieved chunks useful? | > 0.7 | Filters, reranking |
| **Context Recall** | Were all relevant chunks found? | > 0.7 | Increase k, smaller chunks |

### Quick RAGAS Code Pattern

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# 1. Prepare data
eval_data = {
    "question": questions,           # list of strings
    "answer": rag_answers,           # list of strings
    "contexts": retrieved_contexts,  # list of list of strings
    "ground_truth": expected_answers,# list of strings
}

# 2. Create dataset
dataset = Dataset.from_dict(eval_data)

# 3. Evaluate
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
)

# 4. View results
print(result)                # Overall scores
df = result.to_pandas()      # Per-question scores
```

### The Optimization Cycle

```
Build RAG (L10) → Evaluate (L11) → Identify Issues → Fix → Re-evaluate
                         ↑                                           |
                         |___________________________________________|
                              (continuous improvement loop)
```

---

## 14. Key Takeaways

1. ✅ **RAG has 2 failure modes:** bad retrieval OR bad generation — evaluate both separately
2. ✅ **RAGAS automates evaluation** — uses LLMs to score, no manual labeling needed
3. ✅ **Faithfulness is the #1 metric** — if your RAG is hallucinating, fix this first
4. ✅ **Target scores:** faithfulness > 0.8, all others > 0.7
5. ✅ **The optimization cycle:** measure, identify weakest metric, fix, re-measure
6. ✅ **4 inputs needed:** question, answer, contexts, ground_truth

---

## 15. Mini Challenges

### Challenge 1: Add More Test Questions
Add 5 more questions to `test_questions` and `ground_truths`. Include at least one question whose answer is **NOT** in the document (e.g., "What is quantum computing?"). Run RAGAS again and compare scores.

### Challenge 2: Improve Your Weakest Metric
Look at your diagnosis results. Apply the recommended fix (e.g., change k, modify the prompt, adjust chunk_size). Rebuild the chain and run RAGAS again. Did the score improve?

### Challenge 3: Temperature Comparison
Create two RAG chains — one with `temperature=0` and one with `temperature=0.7`. Run RAGAS on both with the same test questions. Compare the faithfulness scores. Which is more grounded?

> 💡 **Hint:** `temperature=0` should have higher faithfulness because it generates more deterministic, grounded answers.

---

## 16. PEP 8 Style Rules Used

| Category | Rule | Example |
|----------|------|---------|
| **Naming** | Variables/functions: `snake_case` | `test_questions`, `ground_truths`, `format_docs()` |
| | Constants: `UPPER_CASE` | `OPENAI_API_KEY` |
| | Classes: `PascalCase` | `ChatOpenAI`, `Dataset`, `StrOutputParser` |
| **Imports** | One import per line | `from ragas import evaluate` |
| | Group: stdlib → third-party | `os` then `ragas` then `langchain` |
| **Whitespace** | Spaces around operators | `score >= 0.8`, `i + 1` |
| | Space after commas | `evaluate(dataset=eval_dataset, metrics=[...])` |
| | Two blank lines before functions | See `format_docs()` |
| **Best Practices** | f-strings for formatting | `f"Score: {score:.4f}"` |
| | `enumerate()` for loops | `for i, question in enumerate(test_questions)` |
| | List comprehensions | `[doc.page_content for doc in retrieved_docs]` |
| | `:<25` formatting | Align columns in output |
| | Docstrings for functions | Triple-quoted descriptions |
| | Dictionary for fixes | Map metric names to fix suggestions |

---

## Appendix: Required Imports Summary

```python
# RAGAS evaluation
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# LangChain components
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Utilities
import os
```

---

**End of Notes** — Hope to Skill: *Building the future, one skill at a time.*