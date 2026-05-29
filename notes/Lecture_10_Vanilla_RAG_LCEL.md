Here are **comprehensive lecture notes** for *Lecture 10: Building Vanilla RAG with LCEL*, based on the provided Jupyter notebook content. These notes cover every concept, important point, code pattern, and best practice.

---

## Lecture 10: Building Vanilla RAG with LCEL
**Course:** NLP with LangChain | **Platform:** Hope to Skill  
**Duration:** ~20 minutes | **Level:** Intermediate  

---

## 1. The Big Picture

**This is it — your first complete RAG system!** Everything from previous lectures comes together here.

**Analogy:** Think of LCEL (LangChain Expression Language) as **plumbing for AI**. You connect pipes together:
- Question flows in
- Retriever finds documents
- Prompt formats everything
- LLM generates the answer
- Clean text flows out

The **`|` pipe operator** connects each stage — just like real plumbing!

---

## 2. What You Will Learn

| # | Topic | Real-World Analogy |
|---|-------|--------------------|
| 1 | What is LCEL? | Plumbing pipes connecting AI components |
| 2 | LCEL vs traditional chains | Modern plumbing vs bucket brigade |
| 3 | The 4 RAG components | Four pipes in your water system |
| 4 | Building the knowledge base | Filling the water tank |
| 5 | Prompt templates | The filter that shapes the output |
| 6 | Assembling the chain | Connecting all the pipes |
| 7 | Running & debugging | Turning on the faucet |
| 8 | Streaming responses | Water flowing continuously |

> 🎯 **By end of this lecture:** You'll have a working end-to-end RAG system in ~20 lines of code. Module 2 is complete!

---

## 3. What Is LCEL? (LangChain Expression Language)

LCEL is a **declarative way to compose AI pipelines** — think of it as connecting Lego blocks to build something bigger.

### The Magic: The Pipe Operator `|`

The `|` operator chains components together. The output of one step becomes the input of the next — just like Unix pipes:

| Unix | LCEL |
|------|------|
| `cat file.txt \| grep "error" \| sort \| head -5` | `retriever \| format_docs \| prompt \| llm \| parser` |

### Comparison: Without LCEL vs With LCEL

| Aspect | Without LCEL | With LCEL |
|--------|-------------|-----------|
| Create components | Same | Same |
| Connect them | Write glue code | Just use `\|` |
| Error handling | Manual | Built-in |
| Streaming | Add later | Built-in |

> 📌 **Key insight:** Each `|` is one step. You can read the chain left to right and understand exactly what happens at each stage.

### Simple LCEL Example (Without RAG)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Component 1: Prompt template
prompt = ChatPromptTemplate.from_template(
    "Tell me one interesting fact about {topic} in 2 sentences."
)

# Component 2: LLM (ChatGPT)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Component 3: Output parser
parser = StrOutputParser()

# Connect them with | (LCEL pipe operator)
simple_chain = prompt | llm | parser

# Run the chain
result = simple_chain.invoke({"topic": "natural language processing"})
```

**What happens when we call `.invoke()`:**
1. `prompt` receives `{"topic": "NLP"}` → produces formatted message
2. `|` passes that message to the next component
3. `llm` receives message → sends to OpenAI → gets response
4. `|` passes response to the next component
5. `parser` receives AI response → extracts just the text string

> 📌 **Key lesson:** Each `|` is a pipe, and data flows through from left to right. No glue code, no boilerplate.

---

## 4. LCEL vs Traditional Chains

Before LCEL, building chains was verbose and hard to debug. LCEL makes everything cleaner:

| Feature | Traditional Chains | LCEL |
|---------|-------------------|------|
| Readability | Nested classes, hard to follow | Read left to right with `\|` |
| Swapping components | Rewrite multiple lines | Change one component |
| Streaming | Manual implementation | Built-in with `.stream()` |
| Async support | Manual with asyncio | Built-in with `.ainvoke()` |
| Error handling | Try/except everywhere | Built-in fallbacks |
| Debugging | Print statements | Inspect each step separately |

### Example Comparison

```python
# Traditional (verbose — old approach)
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
result = chain.run(topic="NLP")

# LCEL (clean — modern approach!)
chain = prompt | llm | parser
result = chain.invoke({"topic": "NLP"})
```

> 🥇 **Rule of thumb:** If you see `LLMChain`, `RetrievalQA`, or other legacy chain classes in tutorials, they're using the old approach. **Always prefer LCEL.**

---

## 5. The 4 RAG Components with LCEL

To build a RAG system, we need exactly 4 components connected with `|`:

```
question --> [Retriever] --> [Prompt] --> [LLM] --> [Parser] --> answer
                |              |           |          |
          finds relevant   formats      generates   extracts
           documents      context +    the answer    plain
                         question                    text
```

| # | Component | What It Does | LangChain Class |
|---|-----------|--------------|-----------------|
| 1 | **Retriever** | Question → list of relevant docs | `vectorstore.as_retriever()` |
| 2 | **Prompt Template** | Formats context + question for LLM | `ChatPromptTemplate` |
| 3 | **LLM** | Generates answer from formatted prompt | `ChatOpenAI` |
| 4 | **Output Parser** | Extracts just the text string | `StrOutputParser` |

### The Power of LCEL: Swappable Components

Each component is independent and swappable:
- Want a different LLM? Change **one line**
- Want a different vector store? Change **one line**
- Want a different prompt? Change **one line**

> 📌 **Key insight:** You can swap any component without touching the rest of the chain.

---

## 6. Building the Knowledge Base (Lectures 5-8 Review)

Before we can build the RAG chain, we need a searchable knowledge base. This combines everything from Lectures 5-8:

```
Load (L5) → Split (L6) → Embed (L7) → Store (L8) → Ready for RAG!
```

### Complete Knowledge Base Setup

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Step 1: LOAD (Lecture 5)
loader = TextLoader("data/nlp_article.txt", encoding="utf-8")
documents = loader.load()

# Step 2: SPLIT (Lecture 6)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Step 3 & 4: EMBED + STORE (Lectures 7 & 8)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name="nlp_course",
)
```

**What's new here:**
- `HuggingFaceEmbeddings` wraps our familiar `all-MiniLM-L6-v2` model in a LangChain-compatible interface
- `QdrantVectorStore.from_documents()` does **embedding + storage in a single call** — no manual `.encode()` or `PointStruct` needed!

---

## 7. Building the Prompt Template

The prompt template is where you control the LLM's behavior. A good RAG prompt tells the LLM:
1. Here's the **context** (retrieved documents)
2. Here's the **question**
3. **Stay grounded** — only answer from the context

### The RAG Prompt Formula

```python
from langchain_core.prompts import ChatPromptTemplate

rag_prompt = ChatPromptTemplate.from_template(
    """You are a helpful teaching assistant for an NLP course.
Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
)
```

**The two variables:**
- `{context}` — will be filled with retrieved document chunks
- `{question}` — will be filled with the user's question

> 📌 **Key lesson:** The prompt is your **steering wheel**. Want shorter answers? Add "Be concise." Want bullet points? Add "Format as a bullet list."

---

## 8. Assembling the RAG Chain — The Main Event!

Now we connect all 4 components with the `|` pipe operator. This is where everything comes together!

### The Chain Architecture

```
Question (string)
    |
    v
{"context": retriever | format_docs, "question": passthrough}
    |
    v
rag_prompt (fills {context} and {question})
    |
    v
llm (generates answer)
    |
    v
StrOutputParser() (extracts text)
    |
    v
Answer (string)
```

### Complete LCEL RAG Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Component 1: RETRIEVER
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Helper: FORMAT DOCS (converts Document list to a single string)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Component 2: PROMPT (created above)
# Component 3: LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Component 4: OUTPUT PARSER
parser = StrOutputParser()

# THE CHAIN — connect everything with |
rag_chain = (
    {
        "context": retriever | format_docs,     # Finds docs → joins them
        "question": RunnablePassthrough(),       # Passes question through
    }
    | rag_prompt                                 # Fills template
    | llm                                        # Generates answer
    | parser                                     # Extracts text
)
```

### Breaking Down the First Step

| Piece | What It Does |
|-------|--------------|
| `retriever \| format_docs` | Finds relevant chunks, joins them into one string |
| `RunnablePassthrough()` | Passes the user's question through unchanged |
| `{...}` dict | Creates a dict with `"context"` and `"question"` keys |

> 📌 **Key insight:** `RunnablePassthrough()` is the key — it takes the user's question string and passes it through as-is, while the retriever uses that same question to find relevant docs.

---

## 9. Running the Chain + Inspecting Output

### Basic Invocation

```python
question = "What is Natural Language Processing?"
answer = rag_chain.invoke(question)
print(answer)
```

**What happens behind the scenes:**
1. Retriever embeds the question and searches Qdrant for similar chunks
2. `format_docs` joins retrieved chunks into a context string
3. Prompt formats context + question into a message
4. LLM (ChatGPT) generates an answer grounded in the context
5. Parser extracts the answer as a plain string

### Debugging: Inspecting Retrieved Chunks

```python
# Call retriever directly to see what chunks were found
retrieved_docs = retriever.invoke(question)

for i, doc in enumerate(retrieved_docs):
    print(f"Chunk #{i + 1}: {doc.page_content[:150]}...")
```

**Debugging Rule of Thumb:**
- If the answer is wrong, **check the retrieved chunks first**
- If retrieved chunks are irrelevant → problem is in the **retriever** (chunk size, overlap, embedding model)
- If chunks are relevant but answer is wrong → problem is in the **prompt**

> 📌 **Pro tip:** Always log the retrieved documents during development. It's the fastest way to find and fix RAG issues.

---

## 10. Streaming Responses — Better User Experience

### The Problem

| Without Streaming | With Streaming |
|------------------|----------------|
| Wait 5 seconds... then full answer | First word in 0.5 sec, flows continuously |
| User thinks "is it broken?" | User reads as answer forms |
| Bad for chat interfaces | **Essential** for chat interfaces |

> 📌 **Key insight:** Streaming doesn't make the LLM faster — it just shows tokens as they're generated instead of waiting for the complete response.

### Streaming Example

```python
question = "Explain how BERT works in simple terms."

print("Streaming answer:\n")
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)  # end="" prevents newlines, flush=True shows immediately
```

### When to Use Which Method

| Method | Use When |
|--------|----------|
| `.invoke()` | Need the full answer as a variable (for processing) |
| `.stream()` | Displaying to a user in real-time (chat interfaces) |
| `.ainvoke()` | Async version for web servers |
| `.astream()` | Async streaming for web servers |

> 📌 **Key lesson:** Streaming is essential for any real chat application. Users expect to see the response forming — waiting for the full answer is not acceptable UX.

---

## 11. Hands-On: Complete RAG with Logging & Timing

```python
import time

def ask_rag(question, chain=rag_chain, ret=retriever, show_sources=True):
    """Ask a question and get a RAG-powered answer with source logging."""
    
    print(f"\nQuestion: {question}")
    print("=" * 65)
    
    # Time the entire RAG pipeline
    start_time = time.time()
    answer = chain.invoke(question)
    elapsed_ms = (time.time() - start_time) * 1000
    
    print(f"\nAnswer:\n{answer}")
    print(f"\n--- Latency: {elapsed_ms:.0f} ms ---")
    
    # Optionally show which chunks were retrieved
    if show_sources:
        retrieved_docs = ret.invoke(question)
        print(f"\n--- Sources ({len(retrieved_docs)} chunks retrieved) ---")
        for i, doc in enumerate(retrieved_docs):
            print(f"  [{i + 1}] {doc.page_content[:100]}...")
    
    return answer
```

**How to evaluate your RAG answers:**

| Check | Good Sign | Bad Sign |
|-------|-----------|----------|
| Factual accuracy | Answer matches source document | Answer contains made-up facts |
| Source relevance | Retrieved chunks relate to question | Retrieved chunks are random |
| Groundedness | Answer only uses info from context | Answer adds info not in context |
| Completeness | Answer covers key points | Answer is too vague |

**Fixing bad answers:**
- Wrong chunks retrieved? → Adjust `chunk_size`, `overlap`, or `k`
- Right chunks, wrong answer? → Adjust the prompt template
- Too slow? → Use a smaller/faster model (`gpt-4o-mini` is fastest)

---

## 12. Quick Reference: LCEL RAG Cheat Sheet

### The Complete RAG Chain Pattern

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 2. Format helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 3. Prompt
prompt = ChatPromptTemplate.from_template(
    "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
)

# 4. LLM + Parser
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

# 5. Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

# Run
answer = rag_chain.invoke("Your question here")

# Stream
for chunk in rag_chain.stream("Your question here"):
    print(chunk, end="", flush=True)
```

### Key Settings

| Setting | Value | Why |
|---------|-------|-----|
| `temperature` | **0** | Factual, consistent answers (no creativity) |
| `k` | 5 | Number of chunks to retrieve (3-10 is typical) |
| `model` | `"gpt-4o-mini"` | Fast and cheap; use `"gpt-4o"` for best quality |
| `chunk_size` | 500 | Good balance for most documents |
| `chunk_overlap` | 50 | Prevents losing context at boundaries |

### The Golden Rules

1. ✅ **`temperature=0` for RAG** — you want factual, grounded answers, not creative storytelling
2. ✅ **Always embed queries with the SAME model** you used for documents (from Lecture 7)

---

## 13. Key Takeaways — Module 2 Complete!

1. ✅ **LCEL = LangChain Expression Language** — connect components with `|`
2. ✅ **4 RAG components:** `retriever | format_docs | prompt | LLM | parser`
3. ✅ **`temperature=0`** for factual, consistent RAG answers
4. ✅ **Streaming (`.stream()`)** is essential for real user-facing applications
5. ✅ **Debug by inspecting retrieved chunks** — most RAG problems are retrieval problems

### The Complete RAG Pipeline

```
Load (L5) → Split (L6) → Embed (L7) → Store (L8) → RAG Chain (L10)
```

### You Now Have a Working RAG System!

Module 2 is complete. You can now:
- Load documents from any source
- Split them intelligently into chunks
- Embed them into meaningful vectors
- Store them in a searchable vector database
- Build an LCEL chain that answers questions from your documents
- Stream responses for great user experience

### What's Next: Module 3

Make it smarter, faster, and more accurate:
- Advanced retrieval strategies (MMR, multi-query)
- Evaluation and testing (RAGAS)
- Production deployment

---

## 14. Mini Challenges

### Challenge 1: Custom Prompt Engineering
Modify the `rag_prompt` template to make the LLM answer in **bullet points** and keep answers under **3 sentences**. Rebuild the chain and test it.

### Challenge 2: Your Own Knowledge Base RAG
Load a different document (text file, PDF, or web page from Lecture 5), build a complete RAG system with LCEL, and ask 3 questions about it.

### Challenge 3: Temperature Experiment
Create two chains — one with `temperature=0` and one with `temperature=0.9`. Ask the same question 3 times on each chain. Which gives more consistent, factual answers?

> 💡 **Hint:** `temperature=0` should give consistent, factual answers. `temperature=0.9` will be more creative but may hallucinate.

---

## 15. PEP 8 Style Rules Used

| Category | Rule | Example |
|----------|------|---------|
| **Naming** | Variables/functions: `snake_case` | `rag_chain`, `format_docs()`, `ask_rag()` |
| | Constants: `UPPER_CASE` | `OPENAI_API_KEY` |
| | Classes: `PascalCase` | `ChatOpenAI`, `ChatPromptTemplate` |
| **Imports** | One import per line | `from langchain_openai import ChatOpenAI` |
| | Group: stdlib → third-party | `os, time` then `langchain` |
| **Whitespace** | Spaces around operators | `i + 1`, `elapsed_ms > 0` |
| | Space after commas | `ChatOpenAI(model="gpt-4o-mini", temperature=0)` |
| | Two blank lines before functions | See `format_docs()`, `ask_rag()` |
| **Best Practices** | f-strings for formatting | `f"Latency: {elapsed_ms:.0f} ms"` |
| | `enumerate()` for loops | `for i, doc in enumerate(retrieved_docs)` |
| | Generator expressions | `"\n\n".join(doc.page_content for doc in docs)` |
| | `:.0f` formatting | Control decimal places |
| | Docstrings for functions | Triple-quoted descriptions |
| | Default parameters | `ask_rag(question, show_sources=True)` |

---

## Appendix: Required Imports Summary

```python
# LCEL core
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Vector store & embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# LLM
from langchain_openai import ChatOpenAI

# Document loading & splitting
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Utilities
import os
import time
```

---

**End of Notes** — Hope to Skill: *Building the future, one skill at a time.*