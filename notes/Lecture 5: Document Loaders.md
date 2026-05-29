Here are **comprehensive lecture notes** for *Lecture 5: Document Loaders — Ingesting Data*, based on the provided Jupyter notebook content. These notes cover every concept, important point, code pattern, and best practice.

---

## Lecture 5: Document Loaders — Ingesting Data
**Course:** NLP with LangChain  
**Platform:** Hope to Skill  
**Duration:** ~20 minutes | **Level:** Intermediate  

---

## 1. The Big Picture (Real-World Motivation)

**Problem:**  
An AI assistant for a bookstore needs data from many places:
- Product catalog → **CSV file**
- Book summaries → **PDF files**
- Author bios → **Websites**
- Customer reviews → **Text files**

**Solution:**  
**Document Loaders** — tools that ingest data from any source into a standardized format.

> ⚡ **Key Fact:** LangChain has **200+ document loaders**, and they all produce the **same output format**. Learn one, and you know them all.

---

## 2. What You Will Learn

| # | Topic | Real-World Analogy |
|---|-------|--------------------|
| 1 | Document object | The "universal shipping box" for all data |
| 2 | Text & PDF loaders | Reading books and notes |
| 3 | CSV loader | Reading spreadsheets row by row |
| 4 | Web loader | Grabbing info from websites |
| 5 | Cloud loaders | Connecting to Google Drive, YouTube, etc. |

---

## 3. Environment Setup

**Required packages:**
```bash
pip install langchain langchain-community pypdf beautifulsoup4 lxml
```

> 💡 **Note:** Run installation once, then you can skip it.

---

## 4. The Document Object — Universal Data Container

### Structure
```
+----------------------------------+
|        Document Object           |
+----------------------------------+
|  page_content = "actual text"    |  ← The item inside the box
|  metadata     = {source, page..} |  ← The shipping label
+----------------------------------+
```

### Attributes
- **`page_content`** (str) — The actual extracted text
- **`metadata`** (dict) — Extra info: source, page number, author, etc.

### Why This Matters
Because every loader outputs the same `Document` box, you can swap a PDF loader for a web loader without changing your pipeline.

### Manual Creation (for understanding)
```python
from langchain_core.documents import Document

sample_document = Document(
    page_content="LangChain makes it easy to build AI apps.",
    metadata={"source": "manual_creation", "page": 0, "author": "Hope to Skill"},
)
```

### Accessing Metadata Safely
```python
# ✅ Safer — returns default if key missing
source = sample_document.metadata.get("source", "unknown")

# ❌ Risky — crashes if key doesn't exist
source = sample_document.metadata["source"]
```

> 🧠 **Key takeaway:** `.get("key", default)` is safer than `["key"]`.

---

## 5. TextLoader — Reading Plain Text Files

**Best for:** `.txt` files, log files, markdown files, any plain text.

**Behavior:** Reads the entire file as **one Document**.

```python
from langchain_community.document_loaders import TextLoader

text_loader = TextLoader(file_path="data/sample.txt", encoding="utf-8")
text_documents = text_loader.load()  # Returns a list of Documents

# Access content
print(text_documents[0].page_content)
print(text_documents[0].metadata)  # {'source': 'data/sample.txt'}
```

**Key pattern (same for all loaders):**
```python
loader = SomeLoader(source)  # Step 1: Create loader
docs = loader.load()         # Step 2: Load → returns list of Documents
```

---

## 6. PyPDFLoader — Reading PDFs Page by Page

**Best for:** Books, research papers, reports, slide decks as PDF.

**Behavior:** Each page becomes a **separate Document**.

```python
from langchain_community.document_loaders import PyPDFLoader

pdf_loader = PyPDFLoader(file_path="data/sample.pdf")
pdf_documents = pdf_loader.load()

print(f"Total pages: {len(pdf_documents)}")  # One Document per page
```

**Metadata includes:**
- `source` — file path
- `page` — page number (starts at 0)
- `total_pages`
- `page_label` — human-readable page number
- PDF-specific fields: `producer`, `creator`, `author`, `title`, etc.

### Comparison: TextLoader vs PyPDFLoader

| Loader | # of Documents | Why? |
|--------|---------------|------|
| TextLoader | 1 | Entire file = 1 document |
| PyPDFLoader | N (pages) | Each page = 1 document |

---

### 📚 PDF Loader Options (Important Reference)

| Loader | Underlying Library | Best For | Speed | Accuracy |
|--------|-------------------|----------|-------|----------|
| PyPDFLoader | pypdf | Simple text PDFs | Fast | Good |
| PyPDFium2Loader | pypdfium2 | Large batches, speed-critical | **Fastest** | Good |
| PyMuPDFLoader | PyMuPDF/fitz | Complex PDFs (tables, images, formatting) | Fast | **Best** |
| PDFPlumberLoader | pdfplumber | PDFs with tables | Medium | **Excellent for tables** |
| UnstructuredPDFLoader | unstructured | Scanned PDFs (OCR), mixed layouts | Slow | Great for messy PDFs |
| AmazonTextractPDFLoader | AWS Textract | Scanned docs at scale (cloud OCR) | Slow | Excellent (paid) |

### Decision Tree for PDF Loaders
```
Is your PDF just plain text?
  └─ YES → PyPDFLoader (simplest)

Does it have complex tables?
  └─ YES → PDFPlumberLoader or PyMuPDFLoader

Is it a scanned image (no selectable text)?
  └─ YES → UnstructuredPDFLoader (with OCR) or AmazonTextractPDFLoader

Need maximum speed for 1000+ PDFs?
  └─ YES → PyPDFium2Loader

Working with Arabic, Chinese, Hindi, etc.?
  └─ YES → PyMuPDFLoader (best Unicode support)
```

> 🔗 Browse all loaders: [LangChain Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/)

---

## 7. CSVLoader — Turning Spreadsheet Rows into Documents

**Best for:** Product catalogs, customer lists, inventory data, survey results.

**Behavior:** Each row becomes a **separate Document**. Column names become part of the content.

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

csv_loader = CSVLoader(file_path="data/sample_products.csv", encoding="utf-8")
csv_documents = csv_loader.load()

# Each document's page_content contains: product_id: 1\nproduct_name: Python Crash Course\n...
# Metadata: {'source': 'data/sample_products.csv', 'row': 0}
```

**Key difference from PDFs:**
- PDF metadata has `page`
- CSV metadata has `row`

---

## 8. WebBaseLoader — Grabbing Data from Websites

**Best for:** Documentation sites, blog posts, Wikipedia articles, news.

**Limitation:** Not great for JavaScript-heavy sites (SPAs) — content may not load.

```python
from langchain_community.document_loaders import WebBaseLoader

# Single page
web_loader = WebBaseLoader(web_paths=["https://en.wikipedia.org/wiki/Natural_language_processing"])
web_documents = web_loader.load()  # One Document per URL

# Multiple pages at once
multi_loader = WebBaseLoader(web_paths=[
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
])
multi_docs = multi_loader.load()
```

**Metadata includes:**
- `source` — URL
- `title` — page title
- `language` — detected language

> ⚠️ **Note:** Web content includes navigation menus, footers, etc. — cleaning comes later.

---

## 9. SitemapLoader — Crawling Entire Websites

**Best for:** Building a knowledge base from docs sites, company wikis.

**Behavior:** Reads `sitemap.xml` and loads every page as a Document.

```python
# Requires nest_asyncio for Jupyter notebooks
import nest_asyncio
nest_asyncio.apply()

from langchain_community.document_loaders.sitemap import SitemapLoader

sitemap_loader = SitemapLoader(
    web_path="https://docs.python.org/3/sitemap.xml",
    filter_urls=["https://docs.python.org/3/tutorial"],  # Limits scope
)
sitemap_docs = sitemap_loader.load()
```

> ⚠️ **Warning:** Can be very slow for large sites — always use `filter_urls`.

---

## 10. Cloud & Special Loaders (Reference)

| Loader | What It Reads | Requirement |
|--------|--------------|-------------|
| `YoutubeLoader` | Video transcripts | `youtube-transcript-api` |
| `GoogleDriveLoader` | Google Docs, Sheets | OAuth credentials |
| `NotionLoader` | Notion pages & databases | Notion API token |
| `S3DirectoryLoader` | Files from AWS S3 | AWS credentials |

**Example pattern (YouTube):**
```python
from langchain_community.document_loaders import YoutubeLoader

loader = YoutubeLoader.from_youtube_url(
    url="https://www.youtube.com/watch?v=VIDEO_ID",
    add_video_info=True,
)
docs = loader.load()
```

> 💡 **Power:** Build a chatbot that pulls from Google Drive docs, Notion wiki, YouTube training videos, and S3 storage — all at once!

---

## 11. Hands-On: PDF Explorer Function

```python
def explore_pdf(file_path):
    """Load a PDF and print a nice summary of its contents."""
    loader = PyPDFLoader(file_path=file_path)
    documents = loader.load()
    
    print(f"PDF: {file_path}")
    print(f"Total pages: {len(documents)}")
    
    # Show first 3 pages (or fewer if PDF is smaller)
    pages_to_show = min(3, len(documents))
    
    for i in range(pages_to_show):
        doc = documents[i]
        print(f"\n--- Page {i + 1} ---")
        print(f"Content preview: {doc.page_content[:300]}")
        print(f"Metadata: {doc.metadata}")
    
    # Show available metadata keys
    if documents:
        print(f"\nAvailable metadata keys: {list(documents[0].metadata.keys())}")
    
    return documents
```

**Techniques used:**
- `min(3, len(documents))` — prevents crash if PDF has < 3 pages
- `documents[0].metadata.keys()` — inspect available metadata
- Returns documents for later use in pipeline

---

## 12. Loader Showdown: Web vs CSV

**Key demonstration:** Despite completely different sources, both return the same `Document` type.

```python
# Load from web and CSV
web_docs = WebBaseLoader(web_paths=["https://en.wikipedia.org/wiki/NLP"]).load()
csv_docs = CSVLoader(file_path="data/sample_products.csv").load()

# Both have the same structure
print(type(web_docs[0]))   # <class 'langchain_core.documents.base.Document'>
print(type(csv_docs[0]))   # <class 'langchain_core.documents.base.Document'>

# Check attributes
hasattr(web_docs[0], 'page_content')  # True
hasattr(web_docs[0], 'metadata')      # True
```

**Differences:**
- What's inside `page_content` (full web page vs one CSV row)
- What's inside `metadata` (URL/title vs file path/row number)

> 🧠 **The whole point:** No matter where data comes from, downstream code (splitting, embeddings, search) stays the same.

---

## 13. Quick Reference: Choosing the Right Loader

### The Golden Rule
**Don't memorize loaders — learn how to FIND the right one.**

Ask yourself:
1. What type of file/source is it? (PDF, CSV, web, cloud, database...)
2. What's special about it? (Tables? Scanned? Non-English? JavaScript-heavy?)
3. Search: `langchain [your source type] loader`

### Common Loaders Reference

| Data Source | Default Loader | When to Use Alternatives |
|-------------|---------------|--------------------------|
| `.txt` file | `TextLoader` | Large files → consider chunking |
| `.pdf` (simple text) | `PyPDFLoader` | Has tables → `PDFPlumberLoader`; Scanned → `UnstructuredPDFLoader` |
| `.pdf` (complex) | `PyMuPDFLoader` | Best overall quality |
| `.csv` file | `CSVLoader` | Big CSVs → pandas + custom loader |
| `.docx` / `.pptx` | `UnstructuredFileLoader` | Needs `pip install unstructured` |
| `.html` file | `UnstructuredHTMLLoader` | Or `BSHTMLLoader` for simpler parsing |
| Web page | `WebBaseLoader` | JavaScript-heavy → `PlaywrightURLLoader` |
| Entire website | `SitemapLoader` | Slow — use `filter_urls` |
| Google Drive | `GoogleDriveLoader` | Needs OAuth setup |
| YouTube video | `YoutubeLoader` | Needs `youtube-transcript-api` |
| Notion | `NotionDirectoryLoader` | Needs API token or exported data |
| AWS S3 | `S3DirectoryLoader` | Needs AWS credentials |
| JSON file | `JSONLoader` | Use `jq_schema` to extract specific fields |
| Markdown | `UnstructuredMarkdownLoader` | Preserves heading structure |
| Email (`.eml`) | `UnstructuredEmailLoader` | Extracts subject, body, attachments |

### Web Loading Options

| Problem | Default | Better Alternative |
|---------|---------|-------------------|
| Static web page | `WebBaseLoader` | Works great! |
| JavaScript-rendered page | `WebBaseLoader` (fails) | `PlaywrightURLLoader` or `SeleniumURLLoader` |
| Need specific HTML elements | `WebBaseLoader` (too much text) | `WebBaseLoader` with `bs_kwargs` to filter tags |
| Crawl entire site | `SitemapLoader` | `RecursiveUrlLoader` if no sitemap exists |

> 💡 **Mindset:** When a loader doesn't give good results, don't force it — search for an alternative.

---

## 14. Key Takeaways

1. ✅ Every loader returns **standardized `Document` objects** — swap loaders without changing code.
2. ✅ Every `Document` has:
   - `page_content` (the text)
   - `metadata` (the label)
3. ✅ Always **inspect your data after loading** — quality and metadata vary by source.
4. ✅ The pattern is **always the same**:
   ```python
   loader = SomeLoader(source)  # Point to your data
   docs = loader.load()         # Get back a list of Documents
   ```
5. ✅ **Next lecture:** Text Splitters — documents are loaded but too big to process at once.

---

## 15. PEP 8 Style Rules Used (Python Best Practices)

| Category | Rule | Example |
|----------|------|---------|
| **Naming** | Variables/functions: `snake_case` | `text_loader`, `explore_pdf()` |
| | Classes: `PascalCase` | `Document`, `TextLoader` |
| **Imports** | One import per line | `from x import TextLoader` |
| | Group: stdlib → third-party → local | (third-party only in this notebook) |
| **Whitespace** | Spaces around operators | `index + 1` not `index+1` |
| | Space after commas | `min(3, len(documents))` |
| | Two blank lines before function definitions | See `explore_pdf()` |
| | Max line length 79 | Slice long strings: `[:300]` |
| **Best Practices** | f-strings for formatting | `f"Pages: {len(docs)}"` |
| | `enumerate()` for loops | `for index, doc in enumerate(docs)` |
| | `.get(key, default)` for dicts | `metadata.get("source", "unknown")` |
| | Docstrings for functions | Triple-quoted description |
| | Descriptive variable names | `pdf_loader` not `pl` |

---

## 16. Mini Challenges (For Practice)

### Challenge 1: The Text Detective
Create a `.txt` file, load with `TextLoader`, print content and metadata.

### Challenge 2: Metadata Inspector
Load `sample.pdf` and print only `source` and `page` from each document's metadata.

### Challenge 3: The Loader Mixer
Load from text, PDF, and CSV into one combined list:
```python
all_docs = text_docs + pdf_docs + csv_docs
```
Print total documents and source of each.

---

## Appendix: Important Notes from the Notebook

- **`%pip install`** — Jupyter magic command to install packages in the notebook environment.
- **`nest_asyncio.apply()`** — Required for `SitemapLoader` in Jupyter to handle event loop conflicts.
- **`enumerate()`** — Gives both index and value in a loop.
- **Slicing `[:200]`** — Shows only first 200 characters as preview.
- **`{content_length:,}`** — Adds commas to large numbers (e.g., `85,432`).
- **`min(3, len(documents))`** — Prevents crashes when list has fewer than expected items.

---

**End of Notes** — Hope to Skill: *Building the future, one skill at a time.*