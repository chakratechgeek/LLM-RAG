# RAG (Retrieval-Augmented Generation) - Comprehensive Guide

## Table of Contents
1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Architecture](#architecture)
4. [Setup & Installation](#setup--installation)
5. [Implementation Guide](#implementation-guide)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Topics](#advanced-topics)
9. [References](#references)

---

## Overview

### What is RAG?

**RAG (Retrieval-Augmented Generation)** combines:
- **Retrieval**: Finding relevant information from your knowledge base
- **Generation**: Using an LLM to generate answers based on retrieved context

### The Problem RAG Solves

**Without RAG:**
```
User: "What's our ExaCC backup policy?"
LLM: *Hallucinates* or "I don't have access to your data"
```

**With RAG:**
```
User: "What's our ExaCC backup policy?"
System: 1. Searches your docs → Finds backup policy
        2. Sends policy to LLM as context
        3. LLM answers from YOUR data
Result: "Your backup policy uses RMAN with 7-day retention..."
```

### Key Benefits

- ✅ **No fine-tuning required** - Use generic LLMs with your data
- ✅ **Always up-to-date** - Update knowledge base, not the model
- ✅ **Reduces hallucination** - Answers grounded in your documents
- ✅ **Cost-effective** - Cheaper than training custom models
- ✅ **Source attribution** - Know which documents provided the answer

---

## Core Concepts

### 1. The RAG Pipeline

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Convert to Embedding│  (Vector representation)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Vector DB Search    │  (Find similar chunks)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Retrieve Top-K      │  (Get 3-5 most relevant chunks)
│ Relevant Chunks     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Build Prompt with   │  (Context + Query)
│ Context             │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ LLM Generates       │  (Answer based on context)
│ Answer              │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Return Answer +     │
│ Sources             │
└─────────────────────┘
```

### 2. Embeddings

**Definition:** Numerical representation of text that captures semantic meaning.

**Example:**
```
Text: "ExaCC requires 3 IPs"
↓ (embedding model)
Vector: [0.23, -0.41, 0.89, ..., 0.15]  # 1536 dimensions
```

**Key Properties:**
- Similar meanings → Close vectors in space
- Different meanings → Distant vectors

**Similarity Measurement:**
```python
"database failure"     → [0.8, 0.3, -0.2]
"DB crash"             → [0.82, 0.28, -0.18]  ← Close (similar)
"pizza recipe"         → [-0.3, -0.7, 0.9]    ← Far (different)
```

**Popular Models:**

| Model | Dimensions | Speed | Quality | Cost |
|-------|-----------|-------|---------|------|
| `text-embedding-ada-002` (OpenAI) | 1536 | Medium | Excellent | $0.0001/1K tokens |
| `all-MiniLM-L6-v2` (Open) | 384 | Fast | Good | Free (local) |
| `bge-large-en` (Open) | 1024 | Medium | Very Good | Free (local) |

### 3. Vector Database

**Purpose:** Store embeddings and perform fast similarity search.

**How It Works:**
```
1. Indexing Phase (One-time):
   Document → Chunks → Embeddings → Store in Vector DB

2. Query Phase (Runtime):
   Query → Embedding → Search Vector DB → Return similar chunks
```

**Internal Structure:**
```
Vector DB Entry:
{
  id: "chunk_42",
  embedding: [0.1, 0.2, ..., 0.9],      # 1536 floats
  document: "Original text chunk",       # Source text
  metadata: {                            # Filters/organization
    source: "manual.pdf",
    page: 42,
    topic: "networking"
  }
}
```

**Popular Options:**

| Database | Type | Best For | Scale |
|----------|------|----------|-------|
| **ChromaDB** | Embedded/Local | Development, prototypes | <10M vectors |
| **Pinecone** | Managed SaaS | Production, no infra mgmt | Billions |
| **Weaviate** | Self-hosted | Full control, K8s | Millions+ |
| **pgvector** | PostgreSQL ext | Already using PostgreSQL | <1M |

**Distance Metrics:**
- **Cosine Similarity** (most common for text): Measures angle between vectors
- **Euclidean (L2)**: Straight-line distance
- **Dot Product**: Raw similarity

### 4. Chunking

**Why Needed:**
- Embedding models have token limits (e.g., 8,191 tokens)
- Documents are often larger (10k+ words)
- Need to split into manageable chunks

**Key Parameters:**

| Parameter | Typical Value | Purpose |
|-----------|---------------|---------|
| **Chunk Size** | 300-600 words | Target size per chunk |
| **Overlap** | 50-100 words | Preserve context at boundaries |
| **Splitters** | `['\n\n', '\n', '. ']` | Hierarchy of split points |

**Chunking Strategies:**

**1. Fixed-Size (Simple)**
```
Split every N characters/tokens
Pros: Fast, predictable
Cons: Breaks mid-sentence
```

**2. Sentence-Based (Better)**
```
Split by sentences, combine until chunk_size
Pros: Preserves sentence integrity
Cons: Variable chunk sizes
```

**3. Paragraph-Based (Structured Docs)**
```
Use document structure (headings, paragraphs)
Pros: Topical coherence
Cons: Requires well-formatted docs
```

**4. Recursive (Production Standard)**
```
Try paragraphs → lines → sentences → words → chars
Pros: Best semantic preservation
Cons: More complex
```

**Example - Bad vs Good Chunking:**

❌ **Bad (no overlap):**
```
Chunk 1: "ExaCC requires"
Chunk 2: "3 client IPs for redundancy"
→ Neither has complete info
```

✅ **Good (with overlap):**
```
Chunk 1: "ExaCC requires 3 client IPs for redundancy. Each IP..."
Chunk 2:              "3 client IPs for redundancy. Each IP must..."
                       ↑ Overlap preserves context
```

### 5. Prompt Engineering

**Critical for RAG Quality:**

**Basic Template:**
```
Answer using ONLY the context below.

CONTEXT:
{retrieved_chunks}

QUESTION:
{user_query}

If answer not in context, say "Information not available in documentation."

ANSWER:
```

**Key Elements:**

| Element | Purpose | Example |
|---------|---------|---------|
| **Instruction** | What to do | "Answer using ONLY context" |
| **Context** | Retrieved chunks | Numbered sources with metadata |
| **Question** | User query | Original question |
| **Constraints** | What NOT to do | "Don't use external knowledge" |
| **Format** | Output structure | "Cite sources as [1], [2]" |

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG SYSTEM                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐        ┌──────────────┐                  │
│  │  Documents   │───────▶│  Chunking    │                  │
│  │  (.pdf, .txt)│        │  Module      │                  │
│  └──────────────┘        └──────┬───────┘                  │
│                                  │                           │
│                                  ▼                           │
│                         ┌──────────────┐                    │
│                         │  Embedding   │                    │
│                         │  Model       │                    │
│                         └──────┬───────┘                    │
│                                │                             │
│                                ▼                             │
│  ┌─────────────────────────────────────────────┐           │
│  │          Vector Database (ChromaDB)          │           │
│  │  - Stores: embeddings + text + metadata     │           │
│  │  - Index: HNSW for fast search              │           │
│  └─────────────────┬───────────────────────────┘           │
│                    │                                         │
│  ┌─────────────────┴───────────────────────────┐           │
│  │         Retrieval Engine                     │           │
│  │  1. Query → Embedding                        │           │
│  │  2. Vector Search (top-k)                    │           │
│  │  3. Return relevant chunks                   │           │
│  └─────────────────┬───────────────────────────┘           │
│                    │                                         │
│                    ▼                                         │
│  ┌─────────────────────────────────────────────┐           │
│  │         Prompt Builder                       │           │
│  │  - Combines: context + query                │           │
│  │  - Formats for LLM                           │           │
│  └─────────────────┬───────────────────────────┘           │
│                    │                                         │
│                    ▼                                         │
│  ┌─────────────────────────────────────────────┐           │
│  │         LLM (GPT-4, Claude, etc.)           │           │
│  │  - Generates answer from context            │           │
│  └─────────────────┬───────────────────────────┘           │
│                    │                                         │
│                    ▼                                         │
│  ┌─────────────────────────────────────────────┐           │
│  │         Response + Citations                 │           │
│  └─────────────────────────────────────────────┘           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Setup & Installation

### Prerequisites

```bash
# Python 3.8+
python --version

# Environment variable for OpenAI API
export open_api_key="sk-your-key-here"
```

### Install Dependencies

```bash
pip install openai chromadb langchain-text-splitters
```

### Verify Installation

```python
import openai
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("✅ All packages installed")
```

---

## Implementation Guide

### Complete Working Example

```python
"""
Complete RAG Implementation
Covers: Chunking → Embedding → Vector DB → Retrieval → LLM
"""

import os
from openai import OpenAI
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================================
# SETUP
# ============================================================================

api_key = os.getenv("open_api_key")
client_openai = OpenAI(api_key=api_key)
client_chroma = chromadb.PersistentClient(path="./rag_database")

# ============================================================================
# STEP 1: CHUNKING
# ============================================================================

def chunk_documents(documents, chunk_size=500, overlap=50):
    """
    Split documents into chunks
    
    Args:
        documents: List of {"text": str, "source": str, "metadata": dict}
        chunk_size: Target chunk size (characters)
        overlap: Overlap between chunks (characters)
    
    Returns:
        List of {"text": str, "metadata": dict}
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=['\n\n', '\n', '. ', ' ', '']
    )
    
    all_chunks = []
    
    for doc in documents:
        chunks = text_splitter.split_text(doc["text"])
        
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    **doc.get("metadata", {}),
                    "source": doc["source"],
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })
    
    return all_chunks

# ============================================================================
# STEP 2: EMBEDDINGS
# ============================================================================

def get_embedding(text):
    """Generate OpenAI embedding for text"""
    response = client_openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def embed_chunks(chunks):
    """Add embeddings to chunks"""
    print(f"Generating embeddings for {len(chunks)} chunks...")
    
    for i, chunk in enumerate(chunks):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(chunks)}")
        
        chunk["embedding"] = get_embedding(chunk["text"])
    
    return chunks

# ============================================================================
# STEP 3: VECTOR DATABASE
# ============================================================================

def create_collection(collection_name="knowledge_base"):
    """Create or reset ChromaDB collection"""
    
    # Delete if exists
    try:
        client_chroma.delete_collection(collection_name)
    except:
        pass
    
    # Create new
    collection = client_chroma.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    return collection

def index_chunks(chunks, collection):
    """Store chunks in vector database"""
    print(f"Indexing {len(chunks)} chunks...")
    
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks]
    )
    
    print(f"✅ Indexed {collection.count()} chunks")

# ============================================================================
# STEP 4: RETRIEVAL
# ============================================================================

def retrieve_context(query, collection, top_k=3):
    """
    Retrieve relevant chunks for query
    
    Returns:
        {
            "chunks": [text1, text2, ...],
            "metadatas": [meta1, meta2, ...],
            "distances": [dist1, dist2, ...]
        }
    """
    # Embed query
    query_embedding = get_embedding(query)
    
    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    return {
        "chunks": results['documents'][0],
        "metadatas": results['metadatas'][0],
        "distances": results['distances'][0]
    }

# ============================================================================
# STEP 5: PROMPT ENGINEERING
# ============================================================================

def build_rag_prompt(query, context_data):
    """Build optimized RAG prompt"""
    
    chunks = context_data["chunks"]
    metadatas = context_data["metadatas"]
    
    # Format context with sources
    context_parts = []
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas), 1):
        source = meta.get('source', 'Unknown')
        context_parts.append(f"[Source {i}] From {source}:\n{chunk}")
    
    context = "\n\n".join(context_parts)
    
    # Build prompt
    prompt = f"""You are a helpful assistant answering questions based on documentation.

DOCUMENTATION:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
- Answer using ONLY the documentation above
- Cite sources using [Source 1], [Source 2], etc.
- If the answer is not in the documentation, say: "This information is not available in the provided documentation."
- Be concise and specific

ANSWER:"""
    
    return prompt

# ============================================================================
# STEP 6: LLM GENERATION
# ============================================================================

def generate_answer(prompt):
    """Get answer from LLM"""
    response = client_openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content

# ============================================================================
# COMPLETE RAG PIPELINE
# ============================================================================

def rag_query(query, collection_name="knowledge_base", top_k=3, verbose=True):
    """
    Complete RAG pipeline
    
    Args:
        query: User question
        collection_name: ChromaDB collection name
        top_k: Number of chunks to retrieve
        verbose: Print intermediate steps
    
    Returns:
        {
            "answer": str,
            "sources": [chunks],
            "metadata": [metadata]
        }
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
    
    # Get collection
    collection = client_chroma.get_collection(collection_name)
    
    # Retrieve context
    if verbose:
        print("\n1. Retrieving relevant context...")
    context_data = retrieve_context(query, collection, top_k)
    
    if verbose:
        print(f"   Retrieved {len(context_data['chunks'])} chunks")
        for i, (chunk, dist) in enumerate(zip(context_data['chunks'], context_data['distances']), 1):
            similarity = 1 - dist
            print(f"   [{i}] Similarity: {similarity:.3f} - {chunk[:80]}...")
    
    # Build prompt
    if verbose:
        print("\n2. Building prompt...")
    prompt = build_rag_prompt(query, context_data)
    
    # Generate answer
    if verbose:
        print("\n3. Generating answer...")
    answer = generate_answer(prompt)
    
    if verbose:
        print(f"\n{'='*60}")
        print("ANSWER:")
        print(answer)
        print('='*60)
    
    return {
        "answer": answer,
        "sources": context_data["chunks"],
        "metadata": context_data["metadatas"]
    }

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    # Sample documents
    documents = [
        {
            "text": """
ExaCC VM Cluster Networking

Network Requirements:
ExaCC VM cluster requires 3 client IP addresses and 2 backup IP addresses for redundancy.
Each IP must be from the client subnet with proper VLAN tagging.

Bond Configuration:
Network bonding uses active-passive mode for client network.
LACP is not supported on client interfaces.
            """,
            "source": "networking_guide.pdf",
            "metadata": {"section": "VM_Cluster", "version": "2024-Q1"}
        },
        {
            "text": """
ExaCC Backup and Recovery

RMAN Configuration:
Backup configuration uses RMAN with following retention policy:
- Level 0 (full) backups: 7 days retention
- Incremental backups: 3 days retention
- Archive logs: 24 hours retention

Backup Schedule:
Level 0 backups run weekly on Sunday 2 AM.
Incremental backups run daily at 2 AM.
            """,
            "source": "backup_guide.pdf",
            "metadata": {"section": "Backup_Policy", "version": "2024-Q1"}
        }
    ]
    
    # Process documents
    print("Processing documents...")
    
    # 1. Chunk
    chunks = chunk_documents(documents, chunk_size=300, overlap=30)
    print(f"Created {len(chunks)} chunks")
    
    # 2. Embed
    chunks = embed_chunks(chunks)
    
    # 3. Index
    collection = create_collection("knowledge_base")
    index_chunks(chunks, collection)
    
    # 4. Query
    result = rag_query(
        "How many IP addresses does ExaCC VM cluster need?",
        collection_name="knowledge_base",
        top_k=2
    )
    
    # 5. Test another query
    result = rag_query(
        "What is the backup retention policy?",
        collection_name="knowledge_base",
        top_k=2
    )
```

---

## Best Practices

### 1. Chunking

✅ **DO:**
- Use recursive splitting (respects document structure)
- Add 10-20% overlap
- Keep chunks 300-600 words
- Track metadata (source, page, chunk index)

❌ **DON'T:**
- Split mid-sentence
- Make chunks too small (<200 words) or large (>1500 words)
- Forget overlap
- Ignore document structure

### 2. Embeddings

✅ **DO:**
- Use same model for indexing and queries
- Batch embed for efficiency
- Cache embeddings (don't re-embed unnecessarily)

❌ **DON'T:**
- Mix embedding models
- Embed one document at a time (slow)
- Re-embed unchanged documents

### 3. Vector Database

✅ **DO:**
- Use cosine similarity for text
- Add rich metadata for filtering
- Limit search to top-k=3-5 chunks

❌ **DON'T:**
- Retrieve 20+ chunks (too much noise)
- Skip metadata (lose filtering capability)
- Use wrong distance metric

### 4. Prompt Engineering

✅ **DO:**
- Explicitly instruct: "Use ONLY context"
- Require source citations
- Provide fallback for missing info
- Use temperature=0 for factual QA

❌ **DON'T:**
- Give vague instructions
- Allow external knowledge usage
- Overload with 50+ chunks
- Skip "not found" handling

### 5. Performance

**Optimize for:**
- **Latency**: Reduce top_k, cache embeddings
- **Quality**: Better chunking, re-ranking
- **Cost**: Batch operations, use local embeddings

**Monitoring:**
```python
# Track these metrics
metrics = {
    "query_latency_ms": 500,       # Time to answer
    "retrieval_accuracy": 0.85,    # % relevant chunks retrieved
    "answer_quality": 0.90,        # Human eval score
    "cost_per_query": 0.002,       # API costs
}
```

---

## Troubleshooting

### Issue: Poor Retrieval Quality

**Symptoms:**
- Irrelevant chunks retrieved
- Correct chunks not in top-k

**Solutions:**
1. **Adjust chunk size**
   ```python
   # Try smaller chunks for precision
   chunk_size=300  # instead of 600
   ```

2. **Increase overlap**
   ```python
   chunk_overlap=100  # instead of 50
   ```

3. **Use better embedding model**
   ```python
   # Switch from ada-002 to custom fine-tuned model
   ```

4. **Add metadata filtering**
   ```python
   results = collection.query(
       query_embeddings=[...],
       where={"section": "networking"}  # Filter by metadata
   )
   ```

### Issue: LLM Hallucination

**Symptoms:**
- Answers contain info not in context
- Made-up facts

**Solutions:**
1. **Strengthen prompt constraints**
   ```python
   prompt = "Answer STRICTLY from context. Do NOT use external knowledge."
   ```

2. **Use lower temperature**
   ```python
   temperature=0  # More deterministic
   ```

3. **Add citation requirement**
   ```python
   prompt = "Cite every fact as [Source X]"
   ```

### Issue: Slow Performance

**Symptoms:**
- >2 second query latency

**Solutions:**
1. **Reduce top_k**
   ```python
   top_k=3  # instead of 10
   ```

2. **Batch embed during indexing**
   ```python
   # Embed 100 chunks at once
   for i in range(0, len(chunks), 100):
       batch = chunks[i:i+100]
       embeddings = embed_batch(batch)
   ```

3. **Use faster embedding model**
   ```python
   # all-MiniLM-L6-v2 (local, fast)
   # instead of ada-002 (API call)
   ```

### Issue: High Costs

**Symptoms:**
- High OpenAI API bills

**Solutions:**
1. **Cache embeddings**
   ```python
   # Only embed new/changed documents
   ```

2. **Use local embedding models**
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')
   # Free, runs locally
   ```

3. **Reduce LLM calls**
   ```python
   # Cache common queries
   # Use cheaper model (gpt-3.5 vs gpt-4)
   ```

---

## Advanced Topics

### 1. Hybrid Search

Combine vector search + keyword search (BM25):

```python
# Vector search (semantic)
vector_results = collection.query(query_embedding, n_results=10)

# Keyword search (exact match)
keyword_results = bm25_search(query, documents)

# Combine and re-rank
final_results = rerank(vector_results + keyword_results)
```

### 2. Re-ranking

Improve retrieval by re-scoring chunks:

```python
# Step 1: Retrieve 20 candidates (fast, less accurate)
candidates = collection.query(query_embedding, n_results=20)

# Step 2: Re-rank with cross-encoder (slow, accurate)
scores = cross_encoder.predict([(query, chunk) for chunk in candidates])

# Step 3: Keep top 3 after re-ranking
final_chunks = top_k(candidates, scores, k=3)
```

### 3. Query Rewriting

Improve query before retrieval:

```python
# Original query
query = "IPs for VM?"

# Rewrite for better retrieval
rewritten = llm.complete(
    f"Rewrite this query to be more specific: {query}"
)
# Result: "How many IP addresses are required for ExaCC VM cluster?"

# Use rewritten query for search
results = retrieve(rewritten)
```

### 4. Multi-Query

Generate multiple query variants:

```python
# Original
query = "ExaCC backup policy"

# Generate variants
variants = [
    "What is the ExaCC backup retention policy?",
    "How long are ExaCC backups kept?",
    "ExaCC RMAN backup configuration"
]

# Retrieve for all variants, merge results
all_results = []
for variant in variants:
    all_results.extend(retrieve(variant))

unique_results = deduplicate(all_results)
```

### 5. Evaluation

Measure RAG quality:

```python
# Test dataset
test_set = [
    {
        "query": "How many IPs?",
        "expected_answer": "3 client, 2 backup (5 total)",
        "relevant_chunks": ["chunk_42"]
    },
    # ... more test cases
]

# Metrics
def evaluate(test_set):
    retrieval_accuracy = []  # % relevant chunks retrieved
    answer_quality = []       # Semantic similarity to expected
    
    for test in test_set:
        # Retrieval
        retrieved = retrieve(test['query'])
        precision = overlap(retrieved, test['relevant_chunks'])
        retrieval_accuracy.append(precision)
        
        # Answer
        answer = rag_query(test['query'])
        similarity = semantic_similarity(answer, test['expected_answer'])
        answer_quality.append(similarity)
    
    return {
        "retrieval_precision": mean(retrieval_accuracy),
        "answer_quality": mean(answer_quality)
    }
```

---

## References

### Documentation
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)

### Papers
- [Retrieval-Augmented Generation (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [Dense Passage Retrieval (Karpukhin et al., 2020)](https://arxiv.org/abs/2004.04906)

### Tools & Libraries
- **Embeddings**: OpenAI, Cohere, Sentence-Transformers
- **Vector DBs**: ChromaDB, Pinecone, Weaviate, Qdrant, Milvus
- **Chunking**: LangChain, LlamaIndex
- **LLMs**: OpenAI (GPT-4), Anthropic (Claude), Open models (Llama)

### Best Practices Guides
- [RAG Best Practices (Pinecone)](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Chunking Strategies (LlamaIndex)](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/)

---

## Quick Reference Card

### Typical Values

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| Chunk size | 300-600 words | Balance context vs precision |
| Overlap | 50-100 words | 10-20% of chunk size |
| Top-k | 3-5 chunks | More = more noise |
| Temperature | 0 (factual QA) | 0.3-0.7 for creative |
| Embedding model | `text-embedding-ada-002` | Or local `all-MiniLM-L6-v2` |
| Distance metric | Cosine | For text embeddings |

### Common Commands

```bash
# Install
pip install openai chromadb langchain-text-splitters

# Set API key
export open_api_key="sk-..."

# Run RAG
python rag_implementation.py
```

### Error Quick Fixes

| Error | Solution |
|-------|----------|
| ChromaDB migration error | Use `chromadb.PersistentClient(path="...")` |
| Token limit exceeded | Reduce chunk_size or top_k |
| Poor retrieval | Increase overlap, adjust chunk_size |
| Hallucination | Strengthen prompt, use temperature=0 |
| Slow queries | Reduce top_k, cache embeddings |

---

## License

This guide is for educational purposes. Refer to individual tool licenses (OpenAI, ChromaDB, etc.) for production use.

## Contributors

Created as a comprehensive RAG learning resource covering embeddings, vector databases, chunking, and prompt engineering.

---

**Last Updated:** January 2025
**Version:** 1.0

