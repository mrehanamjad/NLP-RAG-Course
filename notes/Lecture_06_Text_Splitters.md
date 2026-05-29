Here are **comprehensive lecture notes** for *Lecture 6: Text Splitters & Chunking Strategies*, based on the provided Jupyter notebook content. These notes cover every concept, important point, code pattern, and best practice.

---

## Lecture 6: Text Splitters & Chunking Strategies
**Course:** NLP with LangChain | **Platform:** Hope to Skill  
**Duration:** ~20 minutes | **Level:** Intermediate  

---

## 1. The Big Picture

**The Problem:** A 100-page PDF becomes one giant wall of text. An LLM can't process that. A search engine can't match against that.

**The Solution:** Chunking — cutting documents into smart, bite-sized pieces.

> 🔥 **Key Insight:** Chunking is one of the most important decisions in any RAG system. Bad chunks = bad retrieval = bad answers, no matter how good your LLM is.

---

## 2. What You Will Learn

| # | Topic | Real-World Analogy |
|---|-------|--------------------|
| 1 | Why we chunk documents | Why you slice a pizza before serving |
| 2 | Chunk size — the Goldilocks problem | Slices too small or too big |
| 3 | Overlap — why chunks share text | Pages in a book that repeat the last sentence |
| 4 | RecursiveCharacterTextSplitter | The smart pizza cutter |
| 5 | Other splitters | Specialized tools for special jobs |
| 6 | Metadata preservation | Never lose track of where a chunk came from |

> 🧠 **Key Insight:** Chunking is where most RAG pipelines silently fail. Master this, and you're ahead of 90% of beginners.

---

## 3. Why Do We Chunk Documents? (The Pizza Analogy)

### Three Reasons We Chunk:

| Reason | Explanation |
|--------|-------------|
| **Context Window Limits** | LLMs can only read a limited amount of text at once. You can't send 500 pages to an LLM. |
| **Retrieval Precision** | Find the exact 2-3 paragraphs that answer a question — not dump 50 pages. |
| **Semantic Coherence** | Each chunk should contain one complete idea. Don't cut in the middle of a sentence. |

### Document Size Reality Check

```python
# Example: 8,039 character article ≈ 2,009 tokens
# 1 token ≈ 4 characters (rough estimate for English)
```

> 📌 **Bottom line:** Bad chunking = bad RAG. No matter how good your LLM is, if you feed it garbage chunks, you get garbage answers.

---

## 4. Chunk Size — The Goldilocks Problem

### What Happens at Different Sizes?

| | Too Small (<200 chars) | Just Right (500-1500 chars) | Too Large (>2000 chars) |
|--|----------------------|----------------------------|------------------------|
| **Problem** | Lost context! Can't understand the idea. | One complete idea per chunk. | Too much noise in search results. |
| **Analogy** | Pizza cut into tiny squares | Pizza cut into proper slices | Half the pizza on one plate |

### Recommended Chunk Sizes by Use Case

| Chunk Size | Characters | Good For |
|------------|-----------|----------|
| Small | 200-500 | FAQ answers, short paragraphs |
| Medium | 500-1000 | General purpose **(good default)** |
| Large | 1000-1500 | Technical docs, detailed explanations |
| Very Large | 1500-2000 | Long-form articles, legal documents |

> 📌 **Key lesson:** There's no single "correct" size — it depends on your data and use case.

---

## 5. Overlap — Why Chunks Should Share Text

### The Problem Without Overlap

```
Chunk 1: "...BERT was released by Google."
Chunk 2: "It achieved state-of-the-art results on 11 NLP tasks."
```

If someone asks "What did BERT achieve?", neither chunk alone has the full answer!

### The Solution: Overlap

Overlap means the end of one chunk is repeated at the start of the next chunk.

**How Much Overlap?** Rule of thumb: **10-20% of your chunk size**

| Chunk Size | Recommended Overlap | Why |
|------------|--------------------|-----|
| 500 chars | 50-100 chars | ~1-2 sentences of context |
| 1000 chars | 100-200 chars | ~2-3 sentences of context |
| 1500 chars | 150-300 chars | ~3-5 sentences of context |

> ⚠️ Too much overlap (>30%) = wastes storage. Too little overlap (0%) = context loss at boundaries.

### How Overlap Works (Visual)

```
Step = chunk_size - overlap  # How far we move forward
```

If `chunk_size=80` and `overlap=30`, `step=50`. Move forward 50 chars, repeat the last 30 in the next chunk.

---

## 6. RecursiveCharacterTextSplitter — The Smart Pizza Cutter

**This is the #1 most commonly used splitter in LangChain — your best default.**

### Why "Recursive"?

It tries to split text at natural boundaries, working through a **priority list**:

```
Priority 1: Split at paragraph breaks ("\n\n")
     ↓ (if chunks still too big)
Priority 2: Split at line breaks ("\n")
     ↓ (if chunks still too big)
Priority 3: Split at spaces (" ")
     ↓ (if chunks still too big)
Priority 4: Split at any character ("")
```

**Analogy:** You'd first split between paragraphs. If a paragraph is still too long, split between sentences. Only as a last resort would you split mid-word.

### Basic Usage

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # maximum characters per chunk
    chunk_overlap=200,   # characters to repeat between chunks
)
chunks = splitter.split_documents(documents)
```

### Important Nuance About Overlap

`chunk_overlap` is a **maximum**, not a guarantee:
- If a clean paragraph break exists → overlap might be 0
- If cut happens mid-paragraph → overlap close to requested value

---

## 7. Other Splitter Types — Specialized Tools for Special Jobs

| Splitter | Best For | How It Splits |
|----------|----------|---------------|
| **RecursiveCharacterTextSplitter** | General text (default) | Paragraphs → lines → spaces → chars |
| **CharacterTextSplitter** | Simple, fast splitting | Splits on a single separator (e.g., `\n\n`) |
| **TokenTextSplitter** | Exact token counts | Splits by tokens, not characters |
| **MarkdownTextSplitter** | Markdown docs, wikis | Splits on `#` headers — preserves structure |
| **RecursiveCharacterTextSplitter.from_language()** | Source code | Splits at function/class boundaries |
| **HTMLSectionSplitter** | HTML pages | Splits on HTML tags like `<h1>`, `<h2>` |

### CharacterTextSplitter

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",    # only split at paragraph breaks
    chunk_size=1000,
    chunk_overlap=200,
)
```

**When to use:** Text with very consistent formatting, when speed matters more than perfect splitting.

### TokenTextSplitter

```python
from langchain_text_splitters import TokenTextSplitter

token_splitter = TokenTextSplitter(
    chunk_size=256,      # 256 tokens per chunk (NOT characters!)
    chunk_overlap=50,    # 50 tokens of overlap
)
```

**Why tokens matter:** LLM APIs charge per token, not per character. 1 token ≈ 4 characters in English (varies by language).

**When to use:** Sending chunks directly to LLM, calculating API costs, working with non-English text.

### MarkdownTextSplitter

```python
from langchain_text_splitters import MarkdownTextSplitter

md_splitter = MarkdownTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
md_chunks = md_splitter.split_documents(md_documents)
```

**When to use:** Documentation sites, README files, wikis, knowledge bases.

### Code Splitter

```python
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=30,
)
code_chunks = python_splitter.split_text(python_code)
```

**Supported languages:** Python, JavaScript, TypeScript, Java, Go, Rust, C, C++, Ruby, Scala, Swift, Markdown, LaTeX, HTML, and more.

### Decision Tree: How to Choose the Right Splitter

```
What kind of content?
  |
  ├─ General text → RecursiveCharacterTextSplitter (default)
  ├─ Markdown/wiki → MarkdownTextSplitter
  ├─ Source code → RecursiveCharacterTextSplitter.from_language()
  ├─ Need exact token counts → TokenTextSplitter
  ├─ HTML pages → HTMLSectionSplitter
  └─ Simple text with clear separators → CharacterTextSplitter
```

> 📌 **Same lesson as Lecture 5:** Don't memorize — learn to pick the right tool for the job.

---

## 8. Metadata Preservation — Never Lose Track of Your Chunks

**When you split a document, each chunk inherits the metadata from the original document.**

### What Gets Preserved?

| Source | Metadata in Each Chunk |
|--------|----------------------|
| PDF | `source` (file), `page` (page number) |
| Web | `source` (URL), `title` |
| CSV | `source` (file), `row` (row number) |
| Text | `source` (file path) |

### Adding Custom Metadata

```python
# Add chunk position information
for index, chunk in enumerate(chunks):
    chunk.metadata["chunk_index"] = index
    chunk.metadata["total_chunks"] = len(chunks)
```

**Why add custom metadata?**
- `chunk_index` lets you reconstruct original order
- Add `section_title`, `timestamp`, `author` — anything useful
- This metadata travels with the chunk into the vector store and back to the user

> 🥇 **Golden rule:** Never lose track of where a chunk came from. Your users will want citations, and your debugging will need source tracing.

---

## 9. Hands-On: Split and Explore Function

```python
def split_and_explore(file_path, chunk_size, chunk_overlap):
    """Load a file, split it, and print a detailed analysis."""
    
    # Load document
    loader = TextLoader(file_path=file_path, encoding="utf-8")
    docs = loader.load()
    original_length = len(docs[0].page_content)
    
    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    
    # Calculate statistics
    chunk_lengths = [len(c.page_content) for c in chunks]
    avg_length = sum(chunk_lengths) / len(chunk_lengths)
    min_length = min(chunk_lengths)
    max_length = max(chunk_lengths)
    
    # Print analysis
    print(f"Settings: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print(f"Original: {original_length:,} characters")
    print(f"Result: {len(chunks)} chunks")
    print(f"Avg: {avg_length:,.0f} chars | Min: {min_length} | Max: {max_length}")
    
    # Preview first 3 chunks
    for i in range(min(3, len(chunks))):
        print(f"\n--- Chunk {i+1} ---")
        print(chunks[i].page_content)
    
    return chunks
```

---

## 10. Chunk Size Showdown (300 vs 1000 vs 2000)

| Config | Chunks | Avg Length | Characteristics |
|--------|--------|-----------|-----------------|
| Small (300) | 45 | 201 chars | Very precise, may lack context |
| Medium (1000) | 10 | 810 chars | Good balance of precision and context |
| Large (2000) | 5 | 1,698 chars | Lots of context, but more noise |

### Use Case Recommendations

| Use Case | Recommended Size | Why |
|----------|-----------------|-----|
| FAQ / Q&A bot | 300-500 | Short, focused answers |
| General RAG chatbot | 500-1000 | Good all-around balance |
| Research / analysis | 1000-1500 | Needs more context per chunk |
| Legal / compliance | 1500-2000 | Full clauses need to stay together |

> 📌 **Key lesson:** There is no single right answer. Experiment with 2-3 sizes and evaluate.

---

## 11. Overlap Trade-Off (chunk_size=1000)

| Overlap | Chunks | Storage Overhead | Trade-off |
|---------|--------|------------------|-----------|
| 0 | 10 | -0.2% | Context loss at boundaries |
| 100 | 10 | +0.8% | Good safety net |
| 300 | 10 | +4.5% | Maximum boundary safety |

**The sweet spot (10-20% overlap)** gives you boundary safety without blowing up storage.

---

## 12. Quick Reference: Choosing Splitter & Settings

### Step 1: Pick Your Splitter

| Content Type | Splitter | Why |
|--------------|----------|-----|
| General text | `RecursiveCharacterTextSplitter` | Smart fallback |
| Markdown/wiki | `MarkdownTextSplitter` | Respects heading structure |
| Source code | `RecursiveCharacterTextSplitter.from_language()` | Respects function boundaries |
| Token-sensitive | `TokenTextSplitter` | Exact token counts |
| HTML pages | `HTMLSectionSplitter` | Splits on HTML tags |
| Simple/fast | `CharacterTextSplitter` | One separator, max speed |

### Step 2: Choose Your Settings

| Use Case | `chunk_size` | `chunk_overlap` | Why |
|----------|-------------|-----------------|-----|
| FAQ bot | 300-500 | 30-50 | Short, focused answers |
| General chatbot | 500-1000 | 100-200 | Balanced |
| Research tool | 1000-1500 | 150-300 | Detailed, context-rich |
| Legal/compliance | 1500-2000 | 200-400 | Full clauses together |

### Step 3: Always Verify (3-Check Routine)

```python
# 1. Check chunk count
print(f"Chunks: {len(chunks)}")

# 2. Check chunk sizes (consistency)
sizes = [len(c.page_content) for c in chunks]
print(f"Avg: {sum(sizes)/len(sizes):.0f}, Min: {min(sizes)}, Max: {max(sizes)}")

# 3. Read a few chunks (do they make sense on their own?)
print(chunks[0].page_content)
```

---

## 13. Key Takeaways

1. ✅ **Chunking is critical** — garbage chunks = garbage RAG, no matter how good the LLM
2. ✅ **Default to `RecursiveCharacterTextSplitter`** with 500-1500 chars and 10-20% overlap
3. ✅ **Use specialized splitters** for markdown, code, and token-sensitive tasks
4. ✅ **Always preserve metadata** — users need citations, you need debugging
5. ✅ **Experiment with chunk sizes** — there's no universal "best" size

### The Pipeline So Far

```
Load (Lecture 5) → Split (Lecture 6) → Embed → Store → Retrieve (coming next!)
```

---

## 14. Mini Challenges

### Challenge 1: The Custom Splitter
Split `data/nlp_article.txt` with `chunk_size=500` and `chunk_overlap=75`. Print chunk count and first 2 chunks.

### Challenge 2: The Boundary Inspector
Split with `overlap=100`, then find and print the overlapping text between chunk 1 and chunk 2.

**Hint:** `end_of_chunk_1 = chunks[0].page_content[-100:]` and `start_of_chunk_2 = chunks[1].page_content[:100]`

### Challenge 3: The Full Pipeline
1. Load `data/sample.pdf` with `PyPDFLoader`
2. Split with `RecursiveCharacterTextSplitter`
3. Add `chunk_index` to each chunk's metadata
4. Print summary: chunk number, page number, first 50 characters

**Expected output:**
```
Chunk 1 | Page 0 | "Introduction to NLP Hope to Skill - NLP Co..."
Chunk 2 | Page 0 | "Topics covered in this course: 1. Text Pro..."
```

---

## 15. PEP 8 Style Rules Used

| Category | Rule | Example |
|----------|------|---------|
| **Naming** | Variables/functions: `snake_case` | `chunk_size`, `split_and_explore()` |
| | Classes: `PascalCase` | `RecursiveCharacterTextSplitter` |
| **Imports** | One import per line | `from langchain_text_splitters import RecursiveCharacterTextSplitter` |
| **Whitespace** | Spaces around operators | `chunk_size - overlap` |
| | Space after commas | `min(3, len(chunks))` |
| | Two blank lines before functions | See `split_and_explore()` |
| **Best Practices** | f-strings for formatting | `f"Chunks: {len(chunks)}"` |
| | `enumerate()` for loops | `for i, chunk in enumerate(chunks)` |
| | List comprehensions | `[len(c.page_content) for c in chunks]` |
| | `min()` for safe limits | `min(3, len(chunks))` |
| | Docstrings for functions | Triple-quoted descriptions |

---

## Appendix: Required Imports Summary

```python
# Core splitters
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
    Language,
)

# Loaders (for demo)
from langchain_community.document_loaders import TextLoader, PyPDFLoader
```

---

**End of Notes** — Hope to Skill: *Building the future, one skill at a time.*