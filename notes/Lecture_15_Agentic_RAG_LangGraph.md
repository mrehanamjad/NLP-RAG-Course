Here are **comprehensive lecture notes** for *Lecture 15: Agentic RAG with LangGraph*, based on the provided Jupyter notebook content. These notes cover every concept, important point, code pattern, and best practice.

---

## Lecture 15: Agentic RAG with LangGraph
**Course:** NLP with LangChain | **Platform:** Hope to Skill  
**Duration:** ~20 minutes | **Level:** Advanced  

---

## 1. The Big Picture

**The Problem:** Vanilla RAG (Lecture 10) works — but it has a fatal flaw: it gets **one shot**. Retrieve, generate, done. No second chances.

**The Analogy:**

| System | Analogy |
|--------|---------|
| **Vanilla RAG** | A student who reads one textbook page and writes the answer immediately — no checking, no re-reading, no asking "did I get the right page?" |
| **Agentic RAG** | A smart student who reads, checks if the page was relevant, re-reads if needed, and verifies the answer before submitting |

> 🔥 **Key Insight:** Agentic RAG = Vanilla RAG + Decision Making + Self-Reflection + Retry Logic

---

## 2. What You Will Learn

| # | Topic | Real-World Analogy |
|---|-------|--------------------|
| 1 | Why Vanilla RAG fails | One-shot exam vs. iterative research |
| 2 | What is LangGraph? | A flowchart that runs code |
| 3 | Core concepts: State, Nodes, Edges | Memory, workers, and decision paths |
| 4 | Agentic RAG architecture (5 nodes) | A team of specialists |
| 5 | Conditional routing | Smart traffic lights |
| 6 | Building the State Schema | The agent's shared clipboard |
| 7 | LangGraph code pattern | The recipe to build any agent |
| 8 | RAG + Web Search fallback | Internal library + internet backup |

---

## 3. The Problem with Vanilla RAG

Vanilla RAG follows a linear pipeline:

```
Question → Retrieve → Generate → Answer (done!)
```

This works for simple questions, but fails for complex ones:

| Problem | What Happens | Example |
|---------|--------------|---------|
| **Bad retrieval** | Wrong chunks retrieved, no recovery | "What is BERT?" retrieves chunks about cooking |
| **Bad answer** | LLM hallucinated, no verification | Answer says 2020 when context says 2017 |
| **Multi-step question** | Needs multiple searches, can only do one | "Compare BERT and GPT" needs info from different sections |
| **No self-awareness** | Does not know when it failed | Confidently returns a wrong answer |

### Vanilla RAG vs Agentic RAG

| Feature | Vanilla RAG | Agentic RAG |
|---------|-------------|-------------|
| Attempts | 1 shot | Multiple retries |
| Self-reflection | None | Evaluates its own answer |
| Routing | Always retrieves | Decides: retrieve or answer directly? |
| Recovery | None | Re-retrieves with refined query |
| Fallback | None | Can search the web if knowledge base fails |

---

## 4. What Is LangGraph?

LangGraph is a framework for building **stateful, multi-step AI agents** as directed graphs. Think of it as a **flowchart that runs code**.

- **Documentation:** https://langchain-ai.github.io/langgraph/
- **GitHub:** https://github.com/langchain-ai/langgraph
- **Built by:** LangChain team

### Why LangGraph?

| Feature | LangChain (LCEL) | LangGraph |
|---------|------------------|-----------|
| Flow | Linear pipeline | Directed graph with loops |
| State | Passed through chain | Shared state dictionary |
| Branching | Limited | Conditional edges (if/else routing) |
| Loops | Not possible | Built-in (re-try, re-evaluate) |
| Best for | Simple Q&A | Complex, multi-step agents |

### Visual: LCEL vs LangGraph

```
LCEL (Linear):       A → B → C → D

LangGraph (Graph):   A → B → C
                         |     |
                         v     v
                         D ← E
                         |        (loop back if needed)
                         v
                        END
```

> 📌 **Key insight:** LangGraph can go backwards and loop — that is what makes agents possible.

---

## 5. Core Concepts of LangGraph

| Concept | What It Is | Analogy |
|---------|------------|---------|
| **StateGraph** | The overall graph structure | A flowchart |
| **Node** | A Python function that does work | A worker at a station |
| **Edge** | A connection between two nodes | An arrow in the flowchart |
| **Conditional Edge** | A connection that depends on a condition | A "traffic light" that routes based on rules |
| **State** | A shared dictionary passed between all nodes | A clipboard everyone can read/write |

---

## 6. Agentic RAG Architecture: 5 Nodes

```
                    +----------+
                    |  START   |
                    +----+-----+
                         |
                         v
                  +------+------+
                  | 1. DECIDE   |  Does this need retrieval?
                  +------+------+
                   /           \
                  v             v
          +-------+---+   +----+------+
          | 2. RETRIEVE|   | DIRECT    |
          +-------+---+   | ANSWER    |
                  |       +----+------+
                  v            |
          +-------+------+    |
          | 3. GRADE DOCS|    |
          +-------+------+    |
           /           \      |
     relevant?     not relevant?
          |              |    |
          v              v    |
   +------+------+  +---+--+ |
   | 4. GENERATE |  | WEB  | |
   +------+------+  |SEARCH| |
          |         +---+--+ |
          v              |    |
   +------+------+      |    |
   | 5. EVALUATE |<-----+    |
   +------+------+           |
    /           \            |
  good?       not good?      |
   |              |          |
   v              v          |
 +---+      Re-retrieve      |
 |END|      (loop back)      |
 +---+                       |
```

### What Each Node Does

| Node | Role | Input | Output |
|------|------|-------|--------|
| **Decide** | Route the question | Question | "retrieve" or "direct_answer" |
| **Retrieve** | Search the knowledge base | Question | List of document chunks |
| **Grade Docs** | Check if retrieved docs are relevant | Docs + Question | Filtered docs |
| **Generate** | Create answer from relevant docs | Filtered docs + Question | Answer string |
| **Evaluate** | Check if the answer is good enough | Answer + Question | "good" or "retry" |

---

## 7. Building the State Schema (The Shared Clipboard)

The **State** is the agent's shared memory. Every node reads from it and writes to it.

```python
from typing import TypedDict, List

class AgentState(TypedDict):
    question: str              # The user's question
    documents: List[str]       # Retrieved document chunks
    answer: str                # The generated answer
    iteration_count: int       # How many retry attempts
    search_type: str           # "vectorstore" or "web_search"
```

### Why State Matters

| Without State | With State |
|---------------|------------|
| Each node works in isolation | All nodes share the same data |
| No memory of previous steps | Full history of what happened |
| Cannot count retry attempts | `iteration_count` prevents infinite loops |
| Cannot track decisions made | `search_type` records the routing decision |

> 📌 **Key lesson:** Clean state management = reliable agent behavior. If your state is messy, your agent will be unpredictable.

---

## 8. Node Functions (The Workers)

Each node is a Python function that takes state and returns an updated state.

### Node 1: Retrieve

```python
def retrieve(state: AgentState) -> dict:
    """Retrieve relevant documents from the knowledge base."""
    docs = retriever.invoke(state["question"])
    doc_texts = [doc.page_content for doc in docs]
    return {"documents": doc_texts, "search_type": "vectorstore"}
```

### Node 2: Grade Documents

```python
grade_prompt = ChatPromptTemplate.from_template(
    """Determine if the following document is relevant to the question.
    Reply with ONLY 'yes' or 'no'.

    Document: {document}
    Question: {question}
    Is this document relevant? (yes/no):"""
)

def grade_documents(state: AgentState) -> dict:
    """Grade retrieved documents for relevance."""
    relevant_docs = []
    grader_chain = grade_prompt | llm | StrOutputParser()
    
    for doc in state["documents"]:
        result = grader_chain.invoke({
            "document": doc,
            "question": state["question"],
        })
        if "yes" in result.strip().lower():
            relevant_docs.append(doc)
    
    return {"documents": relevant_docs}
```

### Node 3: Generate Answer

```python
generate_prompt = ChatPromptTemplate.from_template(
    """Answer the question based ONLY on the following context.
    If the context does not contain the answer, say "I don't have enough information."

    Context: {context}
    Question: {question}
    Answer:"""
)

def generate(state: AgentState) -> dict:
    """Generate an answer from the relevant documents."""
    context = "\n\n".join(state["documents"])
    generate_chain = generate_prompt | llm | StrOutputParser()
    answer = generate_chain.invoke({
        "context": context,
        "question": state["question"],
    })
    return {"answer": answer, "iteration_count": state.get("iteration_count", 0) + 1}
```

### Node 4: Evaluate Answer

```python
evaluate_prompt = ChatPromptTemplate.from_template(
    """Determine if the following answer adequately addresses the question.
    Consider: Is it factual? Is it complete?

    Question: {question}
    Answer: {answer}
    Reply with ONLY 'good' or 'not_good':"""
)

def evaluate_answer(state: AgentState) -> dict:
    """Evaluate if the generated answer is good enough."""
    eval_chain = evaluate_prompt | llm | StrOutputParser()
    result = eval_chain.invoke({
        "question": state["question"],
        "answer": state["answer"],
    })
    # Store quality for routing
    return {"quality": result.strip().lower()}
```

### Node 5: Web Search Fallback

```python
from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(max_results=3)

def web_search(state: AgentState) -> dict:
    """Search the web when the knowledge base has no relevant documents."""
    results = web_search_tool.invoke(state["question"])
    web_docs = [r["content"] for r in results if isinstance(r, dict)]
    return {"documents": web_docs, "search_type": "web_search"}
```

---

## 9. Routing Functions (The Traffic Lights)

Routing functions decide **where to go next** based on state.

```python
MAX_ITERATIONS = 3  # Safety limit

def route_after_grading(state: AgentState) -> str:
    """Decide after grading: generate or web search."""
    if state["documents"] and len(state["documents"]) > 0:
        return "generate"
    else:
        return "web_search"

def route_after_evaluation(state: AgentState) -> str:
    """Decide after evaluation: end or retry."""
    iteration = state.get("iteration_count", 1)
    
    if iteration >= MAX_ITERATIONS:
        return "end"
    
    # Check if answer is good enough
    if state.get("quality") == "good":
        return "end"
    else:
        return "retrieve"  # Loop back!
```

---

## 10. Building the LangGraph

```python
from langgraph.graph import StateGraph, END

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("evaluate", evaluate_answer)
workflow.add_node("web_search", web_search)

# Set entry point
workflow.set_entry_point("retrieve")

# Add edges
workflow.add_edge("retrieve", "grade_documents")

# Conditional edge after grading
workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {"generate": "generate", "web_search": "web_search"},
)

# Web search → generate
workflow.add_edge("web_search", "generate")

# Generate → evaluate
workflow.add_edge("generate", "evaluate")

# Conditional edge after evaluation
workflow.add_conditional_edges(
    "evaluate",
    route_after_evaluation,
    {"end": END, "retrieve": "retrieve"},  # Loop back!
)

# Compile the graph
app = workflow.compile()
```

### Running the Agent

```python
result = app.invoke({
    "question": "What is Natural Language Processing?",
    "documents": [],
    "answer": "",
    "iteration_count": 0,
    "search_type": "",
})

print(f"Answer: {result['answer']}")
print(f"Search type: {result['search_type']}")
print(f"Iterations: {result['iteration_count']}")
```

---

## 11. Test Results Demonstration

### Test 1: Question in Knowledge Base

**Question:** "What is Natural Language Processing?"

**Flow:** `retrieve → grade_documents → generate → evaluate → end`

**Result:** Answer from vector store, 4 relevant chunks, 1 iteration

### Test 2: Question NOT in Knowledge Base

**Question:** "What is the latest version of Python released in 2025?"

**Flow:** `retrieve → grade_documents (no relevant) → web_search → generate → evaluate → end`

**Result:** Answer from web search, 3 web results, 1 iteration

### Test 3: Multiple Questions

| Question | Search Type | Docs | Quality |
|----------|-------------|------|---------|
| "What is the transformer architecture?" | vectorstore | 2 | good |
| "How does BERT differ from GPT?" | web_search | 3 | good |
| "What is LangChain used for?" | vectorstore | 4 | good |

---

## 12. Quick Reference: LangGraph Cheat Sheet

### Setup Pattern

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# 1. Define State
class AgentState(TypedDict):
    question: str
    documents: List[str]
    answer: str

# 2. Define Node functions
def my_node(state: AgentState) -> dict:
    # Do work, return updated state fields
    return {"answer": "some answer"}

# 3. Define Routing functions
def my_router(state: AgentState) -> str:
    if condition:
        return "next_node"
    else:
        return "other_node"

# 4. Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("node_name", my_node)
workflow.set_entry_point("node_name")
workflow.add_edge("node_a", "node_b")           # Always go A → B
workflow.add_conditional_edges(                # If/else routing
    "node_a",
    my_router,
    {"option1": "node_b", "option2": "node_c"},
)

# 5. Compile and Run
app = workflow.compile()
result = app.invoke({"question": "my question"})
```

### Key Concepts Summary

| Concept | Code | Purpose |
|---------|------|---------|
| State | `TypedDict` | Shared memory between nodes |
| Node | `def fn(state) -> dict` | Does work, returns state updates |
| Edge | `add_edge(a, b)` | Always go from A to B |
| Conditional Edge | `add_conditional_edges(...)` | Route based on state |
| Entry Point | `set_entry_point(node)` | Where the graph starts |
| END | `END` constant | Where the graph stops |
| Compile | `workflow.compile()` | Validates and prepares the graph |
| Invoke | `app.invoke(state)` | Runs the graph |

### Useful Links

| Resource | URL |
|----------|-----|
| LangGraph Docs | https://langchain-ai.github.io/langgraph/ |
| LangGraph Tutorials | https://langchain-ai.github.io/langgraph/tutorials/ |
| LangGraph GitHub | https://github.com/langchain-ai/langgraph |
| Tavily API | https://docs.tavily.com/ |

---

## 13. Key Takeaways

1. ✅ **Vanilla RAG is one-shot** — no recovery from bad retrieval or bad answers
2. ✅ **LangGraph turns agents into directed graphs** with state, nodes, and edges
3. ✅ **Conditional edges are the secret** — they enable routing, retries, and fallbacks
4. ✅ **Self-reflection** (grade docs + evaluate answer) dramatically improves quality
5. ✅ **Web search fallback** handles questions outside the knowledge base
6. ✅ **`MAX_ITERATIONS` prevents infinite loops** — always set a safety limit
7. ✅ **State management is everything** — clean state = reliable agent

### The Evolution of RAG

| Lecture | System | Pattern |
|---------|--------|---------|
| Lecture 10 | Vanilla RAG | `retrieve → generate → done` |
| Lecture 15 | Agentic RAG | `retrieve → grade → generate → evaluate → (retry?)` |

### When to Use Which

| Approach | Best For |
|----------|----------|
| **Vanilla RAG** | Simple Q&A, prototyping, low latency |
| **Agentic RAG** | Complex questions, high accuracy, production |

---

## 14. Mini Challenges

### Challenge 1: Add a New Node
Add a `rewrite_query` node that rewrites the question to be more specific when the grading node finds no relevant documents (instead of web search).

**Hint:** Use an LLM to rephrase the question.

### Challenge 2: Stricter Evaluation
Modify the `evaluate_answer` function and the `route_after_evaluation` function so that the agent actually retries when the answer is "not_good". Test with a question that requires multiple attempts.

### Challenge 3: Track All Steps
Add a `steps` field to `AgentState` (a list of strings). Have each node append its name to the list. After the agent finishes, print the full path it took.

---

## 15. PEP 8 Style Rules Used

| Category | Rule | Example |
|----------|------|---------|
| **Naming** | Variables/functions: `snake_case` | `grade_documents`, `route_after_grading` |
| | Constants: `UPPER_CASE` | `MAX_ITERATIONS`, `QDRANT_URL` |
| | Classes: `PascalCase` | `AgentState`, `StateGraph` |
| **Best Practices** | `TypedDict` for type-safe state | `class AgentState(TypedDict)` |
| | Docstrings for every function | `"""Retrieve relevant documents..."""` |
| | f-strings for formatting | `f"Found {len(docs)} documents"` |
| | Constants (no magic numbers) | `MAX_ITERATIONS = 3` |
| | Descriptive names | `route_after_grading` not `route1` |
| | `.get()` with default | `state.get('iteration_count', 0)` |

---

## Appendix: Required Imports Summary

```python
# LangGraph core
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional

# LangChain components
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Web search fallback
from langchain_community.tools.tavily_search import TavilySearchResults

# Utilities
import os
```

---

**End of Notes** — Hope to Skill: *Building the future, one skill at a time.*