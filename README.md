# RAG (Retrieval-Augmented Generation) - Complete Implementation Guide

## Table of Contents
1. [Overview](#overview)
2. [Concept #1: The Problem RAG Solves](#concept-1-the-problem-rag-solves)
3. [Concept #2: Embeddings](#concept-2-embeddings)
4. [Concept #3: Vector Database](#concept-3-vector-database)
5. [Concept #4: Chunking](#concept-4-chunking)
6. [Concept #5: Prompt Engineering](#concept-5-prompt-engineering)
7. [Complete RAG System](#complete-rag-system)
8. [Advanced Topics](#advanced-topics)
9. [Production Deployment](#production-deployment)

---

## Overview

### What is RAG?

RAG = **Retrieval-Augmented Generation**

**The Core Flow:**
```
User Question 
  → Search your knowledge base (vector DB)
  → Retrieve top-k relevant chunks
  → Inject chunks into LLM prompt
  → LLM answers using YOUR data
```

**Benefits:**
- ✅ No model fine-tuning needed
- ✅ Always current (update docs, not model)
- ✅ Reduces hallucination
- ✅ Source attribution
- ✅ Cost-effective

---

## Concept #1: The Problem RAG Solves

### Without RAG

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("open_api_key"))

# Query about YOUR proprietary data
query = "What is our ExaCC backup retention policy?"

# LLM doesn't have your data
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": query}]
)

print(response.choices[0].message.content)
# Output: "I don't have access to your specific backup policies..." 
# OR worse: Hallucinates a generic answer
```

### With RAG

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("open_api_key"))

# YOUR knowledge base
knowledge_base = """
ExaCC Backup Policy:
- Level 0 backups: 7 days retention
- Incremental backups: 3 days retention  
- Archive logs: 24 hours retention
"""

# Query with context
query = "What is our ExaCC backup retention policy?"

prompt = f"""Answer based on this documentation:

{knowledge_base}

Question: {query}

Answer:"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
# Output: "Your ExaCC backup retention policy is:
#          - Level 0 backups: 7 days
#          - Incremental backups: 3 days
#          - Archive logs: 24 hours"
```

### The RAG Enhancement

But your knowledge base has 10,000 documents... how do you find the right one?

**That's where embeddings + vector DB come in.**

---

## Concept #2: Embeddings

### 2.1 What Are Embeddings?

**Embeddings = Converting text into numbers (vectors) that capture meaning.**

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("open_api_key"))

def get_embedding(text):
    """Convert text to embedding vector"""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Example
text = "ExaCC VM cluster requires 3 client IP addresses"
embedding = get_embedding(text)

print(f"Text: {text}")
print(f"Embedding dimension: {len(embedding)}")  # 1536
print(f"First 10 values: {embedding[:10]}")
# Output: [0.0023, -0.0156, 0.0089, -0.0234, ...]
```

### 2.2 Why Embeddings Capture Meaning

```python
from openai import OpenAI
import numpy as np
import os

client = OpenAI(api_key=os.getenv("open_api_key"))

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    """Calculate similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Three texts
text1 = "ExaCC VM cluster network configuration"
text2 = "Virtual machine cluster networking setup"  # Similar meaning
text3 = "How to bake chocolate cake"               # Different meaning

# Get embeddings
emb1 = get_embedding(text1)
emb2 = get_embedding(text2)
emb3 = get_embedding(text3)

# Calculate similarities
sim_1_2 = cosine_similarity(emb1, emb2)
sim_1_3 = cosine_similarity(emb1, emb3)

print(f"Similarity (text1 vs text2): {sim_1_2:.4f}")  # ~0.85 (HIGH - similar)
print(f"Similarity (text1 vs text3): {sim_1_3:.4f}")  # ~0.15 (LOW - different)

# Key insight: Similar meanings → similar vectors → high cosine similarity
```

### 2.3 Batch Embedding (Efficient)

```python
from openai import OpenAI
import os
import time

client = OpenAI(api_key=os.getenv("open_api_key"))

documents = [
    "ExaCC requires 3 client IPs and 2 backup IPs",
    "Backup retention is 7 days for Level 0",
    "Patching takes minimum 2 hour window",
    "Network bonding uses active-passive mode",
    "DNS entries required before cluster creation"
]

# ❌ BAD: One at a time (slow, expensive)
def embed_one_by_one(texts):
    embeddings = []
    start = time.time()
    
    for text in texts:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        embeddings.append(response.data[0].embedding)
    
    elapsed = time.time() - start
    print(f"One-by-one: {elapsed:.2f}s for {len(texts)} texts")
    return embeddings

# ✅ GOOD: Batch (fast, cheaper)
def embed_batch(texts, batch_size=100):
    embeddings = []
    start = time.time()
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=batch  # Send multiple at once
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    
    elapsed = time.time() - start
    print(f"Batch: {elapsed:.2f}s for {len(texts)} texts")
    return embeddings

# Test both
print("Testing embedding methods:")
embed_one_by_one(documents)
embed_batch(documents)

# Batch is ~5-10x faster and costs the same!
```

### 2.4 Embedding Model Comparison

```python
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
import time

client = OpenAI(api_key=os.getenv("open_api_key"))

text = "ExaCC VM cluster requires 3 client IP addresses"

# Option 1: OpenAI (API, paid)
def openai_embedding(text):
    start = time.time()
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    elapsed = time.time() - start
    
    embedding = response.data[0].embedding
    print(f"\nOpenAI ada-002:")
    print(f"  Dimension: {len(embedding)}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Cost: $0.0001 per 1K tokens")
    return embedding

# Option 2: Open Source (Local, free)
def local_embedding(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    start = time.time()
    embedding = model.encode(text)
    elapsed = time.time() - start
    
    print(f"\nall-MiniLM-L6-v2 (local):")
    print(f"  Dimension: {len(embedding)}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Cost: $0 (runs locally)")
    return embedding

# Compare
openai_emb = openai_embedding(text)
local_emb = local_embedding(text)

"""
Output:
OpenAI ada-002:
  Dimension: 1536
  Time: 0.450s
  Cost: $0.0001 per 1K tokens

all-MiniLM-L6-v2 (local):
  Dimension: 384
  Time: 0.025s
  Cost: $0 (runs locally)

Trade-off: OpenAI = better quality, Local = faster + free
"""
```

### 2.5 Embedding Storage & Reuse

```python
from openai import OpenAI
import os
import json
import hashlib

client = OpenAI(api_key=os.getenv("open_api_key"))

class EmbeddingCache:
    """Cache embeddings to avoid re-computing"""
    
    def __init__(self, cache_file="embeddings_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load cached embeddings from disk"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def _get_hash(self, text):
        """Create unique hash for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_embedding(self, text):
        """Get embedding (from cache or API)"""
        text_hash = self._get_hash(text)
        
        # Check cache first
        if text_hash in self.cache:
            print(f"  ✓ Cache hit for: {text[:50]}...")
            return self.cache[text_hash]
        
        # Not in cache - call API
        print(f"  ✗ Cache miss, calling API for: {text[:50]}...")
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        embedding = response.data[0].embedding
        
        # Store in cache
        self.cache[text_hash] = embedding
        self._save_cache()
        
        return embedding

# Usage
cache = EmbeddingCache()

# First call - hits API
emb1 = cache.get_embedding("ExaCC requires 3 IPs")

# Second call - uses cache (free, instant)
emb2 = cache.get_embedding("ExaCC requires 3 IPs")

# Different text - hits API again
emb3 = cache.get_embedding("Backup policy is 7 days")

"""
Output:
  ✗ Cache miss, calling API for: ExaCC requires 3 IPs...
  ✓ Cache hit for: ExaCC requires 3 IPs...
  ✗ Cache miss, calling API for: Backup policy is 7 days...

Savings: 50% fewer API calls in this example!
"""
```

### 2.6 Embedding Visualization (2D Projection)

```python
from openai import OpenAI
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

client = OpenAI(api_key=os.getenv("open_api_key"))

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Documents from different topics
documents = {
    "Networking": [
        "ExaCC requires 3 client IPs",
        "Network bonding uses active-passive",
        "VLAN configuration is mandatory",
    ],
    "Backup": [
        "RMAN retention is 7 days",
        "Incremental backups run daily",
        "Archive logs kept 24 hours",
    ],
    "Patching": [
        "Patching takes 2 hour window",
        "Rolling patches for non-CDB only",
        "Test patches in non-prod first",
    ]
}

# Get embeddings
all_texts = []
all_labels = []
all_embeddings = []

for category, texts in documents.items():
    for text in texts:
        all_texts.append(text)
        all_labels.append(category)
        all_embeddings.append(get_embedding(text))

# Convert to numpy array
embeddings_array = np.array(all_embeddings)

# Reduce from 1536 dimensions to 2D (for visualization)
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_array)

# Plot
plt.figure(figsize=(10, 6))
colors = {'Networking': 'blue', 'Backup': 'green', 'Patching': 'red'}

for i, label in enumerate(all_labels):
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                c=colors[label], label=label if label not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.annotate(all_texts[i][:20], (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                 fontsize=8, alpha=0.7)

plt.legend()
plt.title("Embedding Space (1536D → 2D projection)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.savefig("embeddings_visualization.png")
print("Saved visualization to embeddings_visualization.png")

# Notice: Similar topics cluster together!
```

---

## Concept #3: Vector Database

### 3.1 Basic ChromaDB Setup

```python
import chromadb

# Option 1: In-memory (data lost on exit)
client = chromadb.EphemeralClient()
print("Created in-memory client (temporary)")

# Option 2: Persistent (data saved to disk) ✅ RECOMMENDED
client = chromadb.PersistentClient(path="./my_vector_db")
print("Created persistent client at ./my_vector_db")

# Create collection (like a "table" in SQL)
collection = client.create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)

print(f"Created collection: {collection.name}")
print(f"Collection count: {collection.count()}")  # 0 initially
```

### 3.2 Adding Documents to ChromaDB

```python
import chromadb
from openai import OpenAI
import os

client_openai = OpenAI(api_key=os.getenv("open_api_key"))
client_chroma = chromadb.PersistentClient(path="./vector_db")

def get_embedding(text):
    response = client_openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Create collection
collection = client_chroma.create_collection(
    name="exacc_docs",
    metadata={"hnsw:space": "cosine"}
)

# Documents to index
documents = [
    "ExaCC VM cluster requires 3 client IP addresses and 2 backup IP addresses",
    "Backup retention policy: Level 0 backups kept for 7 days",
    "Patching requires minimum 2-hour maintenance window",
    "Network bonding uses active-passive mode for client network",
    "DNS entries must be configured before VM cluster creation"
]

# Generate embeddings
print("Generating embeddings...")
embeddings = []
for doc in documents:
    emb = get_embedding(doc)
    embeddings.append(emb)
    print(f"  ✓ Embedded: {doc[:50]}...")

# Add to ChromaDB
print("\nIndexing in ChromaDB...")
collection.add(
    ids=[f"doc_{i}" for i in range(len(documents))],  # Unique IDs
    embeddings=embeddings,                             # Vectors
    documents=documents,                               # Original text
    metadatas=[
        {"topic": "networking", "priority": "high"},
        {"topic": "backup", "priority": "medium"},
        {"topic": "patching", "priority": "high"},
        {"topic": "networking", "priority": "medium"},
        {"topic": "provisioning", "priority": "high"}
    ]
)

print(f"✅ Indexed {collection.count()} documents")

"""
Output:
Generating embeddings...
  ✓ Embedded: ExaCC VM cluster requires 3 client IP addresses...
  ✓ Embedded: Backup retention policy: Level 0 backups kept...
  ✓ Embedded: Patching requires minimum 2-hour maintenance...
  ✓ Embedded: Network bonding uses active-passive mode for...
  ✓ Embedded: DNS entries must be configured before VM clus...

Indexing in ChromaDB...
✅ Indexed 5 documents
"""
```

### 3.3 Querying ChromaDB (Similarity Search)

```python
import chromadb
from openai import OpenAI
import os

client_openai = OpenAI(api_key=os.getenv("open_api_key"))
client_chroma = chromadb.PersistentClient(path="./vector_db")

def get_embedding(text):
    response = client_openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Get collection
collection = client_chroma.get_collection("exacc_docs")

# User query
query = "How many IP addresses are needed for VM cluster?"

# Convert query to embedding
print(f"Query: {query}")
query_embedding = get_embedding(query)

# Search for similar documents
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3  # Get top 3 most similar
)

# Display results
print(f"\nTop {len(results['documents'][0])} relevant documents:")
print("="*70)

for i, (doc, distance, metadata) in enumerate(zip(
    results['documents'][0],
    results['distances'][0],
    results['metadatas'][0]
), 1):
    similarity = 1 - distance  # Convert distance to similarity (0-1)
    print(f"\n[{i}] Similarity: {similarity:.4f}")
    print(f"    Topic: {metadata['topic']} | Priority: {metadata['priority']}")
    print(f"    Text: {doc}")

"""
Output:
Query: How many IP addresses are needed for VM cluster?

Top 3 relevant documents:
======================================================================

[1] Similarity: 0.8923
    Topic: networking | Priority: high
    Text: ExaCC VM cluster requires 3 client IP addresses and 2 backup IP addresses

[2] Similarity: 0.7156
    Topic: provisioning | Priority: high
    Text: DNS entries must be configured before VM cluster creation

[3] Similarity: 0.6834
    Topic: networking | Priority: medium
    Text: Network bonding uses active-passive mode for client network

Notice: Most relevant document is ranked #1!
"""
```

### 3.4 Metadata Filtering

```python
import chromadb
from openai import OpenAI
import os

client_openai = OpenAI(api_key=os.getenv("open_api_key"))
client_chroma = chromadb.PersistentClient(path="./vector_db")

def get_embedding(text):
    response = client_openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

collection = client_chroma.get_collection("exacc_docs")

query = "network configuration"
query_embedding = get_embedding(query)

# Example 1: Filter by topic
print("Example 1: Only networking documents")
print("="*60)
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    where={"topic": "networking"}  # Metadata filter
)

for doc in results['documents'][0]:
    print(f"  • {doc[:60]}...")

# Example 2: Filter by priority
print("\nExample 2: Only high priority documents")
print("="*60)
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    where={"priority": "high"}
)

for doc in results['documents'][0]:
    print(f"  • {doc[:60]}...")

# Example 3: Combined filters (AND)
print("\nExample 3: High priority networking documents")
print("="*60)
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    where={
        "$and": [
            {"topic": "networking"},
            {"priority": "high"}
        ]
    }
)

for doc in results['documents'][0]:
    print(f"  • {doc[:60]}...")

# Example 4: OR filter
print("\nExample 4: Networking OR backup documents")
print("="*60)
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    where={
        "$or": [
            {"topic": "networking"},
            {"topic": "backup"}
        ]
    }
)

for doc in results['documents'][0]:
    print(f"  • {doc[:60]}...")

"""
Metadata filtering lets you:
- Search within specific document types
- Filter by date/version/department
- Combine semantic search with structured filters
"""
```

### 3.5 CRUD Operations

```python
import chromadb
from openai import OpenAI
import os

client_openai = OpenAI(api_key=os.getenv("open_api_key"))
client_chroma = chromadb.PersistentClient(path="./vector_db")

def get_embedding(text):
    response = client_openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

collection = client_chroma.get_collection("exacc_docs")

# CREATE - Add new document
print("1. CREATE - Adding new document")
new_doc = "ExaCC patching can be done in rolling fashion for RAC databases"
new_embedding = get_embedding(new_doc)

collection.add(
    ids=["doc_new_1"],
    embeddings=[new_embedding],
    documents=[new_doc],
    metadatas=[{"topic": "patching", "priority": "medium", "added": "2024-01-15"}]
)
print(f"   ✓ Added document. Total count: {collection.count()}")

# READ - Get specific document by ID
print("\n2. READ - Get document by ID")
result = collection.get(
    ids=["doc_new_1"],
    include=["documents", "metadatas", "embeddings"]
)
print(f"   Document: {result['documents'][0]}")
print(f"   Metadata: {result['metadatas'][0]}")

# UPDATE - Modify existing document
print("\n3. UPDATE - Updating document")
updated_doc = "ExaCC patching supports rolling updates for RAC and non-RAC databases"
updated_embedding = get_embedding(updated_doc)

collection.update(
    ids=["doc_new_1"],
    embeddings=[updated_embedding],
    documents=[updated_doc],
    metadatas=[{"topic": "patching", "priority": "high", "updated": "2024-01-16"}]
)
print("   ✓ Updated document")

# Verify update
result = collection.get(ids=["doc_new_1"])
print(f"   New text: {result['documents'][0]}")

# DELETE - Remove document
print("\n4. DELETE - Removing document")
collection.delete(ids=["doc_new_1"])
print(f"   ✓ Deleted document. Total count: {collection.count()}")

# Verify deletion
try:
    result = collection.get(ids=["doc_new_1"])
    if not result['ids']:
        print("   ✓ Confirmed: Document no longer exists")
except:
    print("   ✓ Confirmed: Document no longer exists")
```

### 3.6 Collection Management

```python
import chromadb

client = chromadb.PersistentClient(path="./vector_db")

# List all collections
print("Current collections:")
collections = client.list_collections()
for coll in collections:
    print(f"  • {coll.name} ({coll.count()} documents)")

# Get collection info
collection = client.get_collection("exacc_docs")
print(f"\nCollection '{collection.name}':")
print(f"  Count: {collection.count()}")
print(f"  Metadata: {collection.metadata}")

# Create new collection
new_collection = client.create_collection(
    name="test_collection",
    metadata={"description": "Temporary test collection"}
)
print(f"\n✓ Created: {new_collection.name}")

# Delete collection
client.delete_collection("test_collection")
print("✓ Deleted: test_collection")

# Get or create (idempotent)
collection = client.get_or_create_collection(
    name="exacc_docs",
    metadata={"hnsw:space": "cosine"}
)
print(f"\n✓ Got or created: {collection.name}")
```

---

## Concept #4: Chunking

### 4.1 Why Chunking is Needed

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("open_api_key"))

# Large document (10,000 words)
large_document = """
[... 10,000 words about ExaCC covering networking, backup, patching, etc ...]
"""

# Problem 1: Embedding model has token limit
try:
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=large_document  # Limit: 8191 tokens (~6000 words)
    )
except Exception as e:
    print(f"❌ Error: {e}")
    print("Document exceeds token limit!")

# Problem 2: Even if it fits, single embedding loses specificity
# Query: "How many IPs?" would retrieve entire 10k word document
# LLM gets flooded with irrelevant info

# Solution: CHUNKING
print("\n✅ Solution: Split into chunks")
print("10,000 word doc → 20 chunks of 500 words each")
print("Query retrieves only relevant 1-2 chunks (500-1000 words)")
```

### 4.2 Fixed-Size Chunking

```python
def chunk_fixed_size(text, chunk_size=500, overlap=50):
    """
    Split text into fixed-size chunks with overlap
    
    Args:
        text: Input text
        chunk_size: Characters per chunk
        overlap: Overlapping characters between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        # Get chunk
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move forward (accounting for overlap)
        start = end - overlap
        
        # Stop if we've passed the end
        if start >= len(text):
            break
    
    return chunks

# Example
document = """
ExaCC VM Cluster Configuration

Network Requirements:
ExaCC VM cluster requires 3 client IP addresses and 2 backup IP addresses for redundancy.
Each IP must be from the client subnet with proper VLAN tagging.

Backup Policy:
RMAN backup uses following retention: Level 0 backups kept for 7 days, incremental backups 
for 3 days, and archive logs for 24 hours.

Patching Guidelines:
Patching requires minimum 2-hour maintenance window. Rolling patches available only for 
non-CDB environments. Always test patches in non-production first.
""" * 5  # Repeat to make it longer

# Chunk
chunks = chunk_fixed_size(document, chunk_size=200, overlap=20)

print(f"Original length: {len(document)} characters")
print(f"Number of chunks: {len(chunks)}")
print(f"\nFirst 3 chunks:")
for i, chunk in enumerate(chunks[:3], 1):
    print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
    print(chunk[:100] + "...")
```

### 4.3 Sentence-Based Chunking

```python
import re

def chunk_by_sentences(text, chunk_size=500, overlap=50):
    """
    Split text by sentences, combine until chunk_size
    Preserves sentence boundaries
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If adding this sentence exceeds chunk_size
        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Create overlap: keep last few sentences
            overlap_sentences = []
            overlap_length = 0
            
            for s in reversed(current_chunk):
                if overlap_length + len(s) < overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s)
                else:
                    break
            
            # Start new chunk with overlap
            current_chunk = overlap_sentences
            current_length = overlap_length
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Example
document = """
ExaCC VM cluster requires 3 client IP addresses. Each IP must be from client subnet. 
VLAN tagging is mandatory. DNS entries must exist before creation.

Backup uses RMAN retention policy. Level 0 backups kept 7 days. Incremental backups 
kept 3 days. Archive logs kept 24 hours.

Patching takes minimum 2 hours. Rolling patches for non-CDB only. Test in non-prod first.
"""

chunks = chunk_by_sentences(document, chunk_size=150, overlap=30)

print(f"Created {len(chunks)} chunks\n")
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i} ({len(chunk)} chars):")
    print(chunk)
    print()
```

### 4.4 Recursive Character Splitter (Production Standard)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Target size
    chunk_overlap=50,      # Overlap
    length_function=len,   # How to measure length
    separators=[
        '\n\n',    # Priority 1: Paragraphs
        '\n',      # Priority 2: Lines
        '. ',      # Priority 3: Sentences
        ' ',       # Priority 4: Words
        ''         # Priority 5: Characters
    ]
)

document = """
# ExaCC Configuration Guide

## Networking

### IP Requirements
ExaCC VM cluster requires 3 client IP addresses and 2 backup IP addresses for redundancy.
Each IP must be from the client subnet.

### Network Bonding
Network bonding uses active-passive mode for client network.
LACP is not supported on client interfaces.

## Backup

### RMAN Configuration
Backup configuration uses RMAN with following retention policy:
- Level 0 backups: 7 days
- Incremental backups: 3 days
- Archive logs: 24 hours

### Backup Schedule
Level 0 backups run weekly on Sunday 2 AM.
Incremental backups run daily at 2 AM.
"""

# Split
chunks = text_splitter.split_text(document)

print(f"Created {len(chunks)} chunks\n")

for i, chunk in enumerate(chunks, 1):
    print(f"{'='*60}")
    print(f"Chunk {i} ({len(chunk)} chars):")
    print(chunk)
    print()

"""
How it works:
1. Try splitting by '\n\n' (paragraphs)
2. If chunks still too big, try '\n' (lines)
3. If still too big, try '. ' (sentences)
4. If still too big, try ' ' (words)
5. Last resort: split by characters

This preserves document structure while respecting chunk_size
"""
```

### 4.5 Token-Based Chunking (Matches Embedding Model)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Character-based (default)
char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 500 CHARACTERS
    chunk_overlap=50
)

# Token-based (better for embeddings)
token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="text-embedding-ada-002",  # Uses same tokenizer
    chunk_size=500,                        # 500 TOKENS (not characters)
    chunk_overlap=50
)

document = "ExaCC " * 1000  # Lots of repetition

# Compare
char_chunks = char_splitter.split_text(document)
token_chunks = token_splitter.split_text(document)

print(f"Character-based: {len(char_chunks)} chunks")
print(f"Token-based: {len(token_chunks)} chunks")

print(f"\nWhy different?")
print(f"  'ExaCC' might be:")
print(f"    - 5 characters")
print(f"    - 2-3 tokens (depends on tokenization)")

print(f"\n✅ Use token-based for embeddings!")
print(f"   Ensures chunks never exceed model's token limit (8191 for ada-002)")
```

### 4.6 Chunking with Metadata Preservation

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents_with_metadata(documents, chunk_size=500, overlap=50):
    """
    Chunk multiple documents while preserving metadata
    
    Args:
        documents: List of {"text": str, "metadata": dict}
        chunk_size: Target chunk size
        overlap: Overlap between chunks
    
    Returns:
        List of {"text": str, "metadata": dict} with chunk info
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    
    all_chunks = []
    
    for doc in documents:
        # Split document
        chunks = text_splitter.split_text(doc["text"])
        
        # Add metadata to each chunk
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {
                **doc["metadata"],  # Original metadata
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk_text)
            }
            
            all_chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
    
    return all_chunks

# Example
documents = [
    {
        "text": "ExaCC networking guide content here... " * 100,
        "metadata": {
            "source": "networking_guide.pdf",
            "page": 42,
            "section": "VM_Cluster",
            "author": "DevOps Team",
            "version": "2024-Q1"
        }
    },
    {
        "text": "ExaCC backup policy content here... " * 100,
        "metadata": {
            "source": "backup_guide.pdf",
            "page": 15,
            "section": "RMAN_Config",
            "author": "DBA Team",
            "version": "2024-Q1"
        }
    }
]

# Chunk with metadata
chunks = chunk_documents_with_metadata(documents, chunk_size=200, overlap=20)

print(f"Total chunks: {len(chunks)}\n")

# Show first chunk from each document
for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i+1}:")
    print(f"  Text: {chunk['text'][:60]}...")
    print(f"  Metadata: {chunk['metadata']}")
    print()

"""
Output shows:
- Original metadata (source, page, section, author, version)
- Plus chunk-specific metadata (chunk_index, total_chunks, chunk_size)

This allows:
- Tracing chunks back to source documents
- Filtering by any metadata field
- Citing specific pages in answers
"""
```

### 4.7 Chunking Quality Analysis

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def analyze_chunks(chunks):
    """Analyze chunk quality metrics"""
    
    lengths = [len(c) for c in chunks]
    
    print(f"Chunk Analysis:")
    print(f"{'='*60}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Average size: {sum(lengths)/len(lengths):.0f} chars")
    print(f"  Min size: {min(lengths)} chars")
    print(f"  Max size: {max(lengths)} chars")
    print(f"  Size std dev: {(sum((x - sum(lengths)/len(lengths))**2 for x in lengths) / len(lengths))**0.5:.0f}")
    
    # Check for very small chunks
    small_chunks = [c for c in chunks if len(c) < 100]
    if small_chunks:
        print(f"\n  ⚠️ Warning: {len(small_chunks)} chunks < 100 chars")
        print(f"     Example: '{small_chunks[0][:50]}...'")
    else:
        print(f"\n  ✓ All chunks >= 100 chars")
    
    # Check for sentence fragments
    fragments = [c for c in chunks if not c.strip().endswith(('.', '!', '?', '\n', ':'))]
    if fragments:
        print(f"  ⚠️ Warning: {len(fragments)} chunks end mid-sentence")
        print(f"     Example: '{fragments[0][-50:]}'")
    else:
        print(f"  ✓ All chunks end at sentence boundaries")
    
    # Check overlap
    if len(chunks) > 1:
        overlaps = []
        for i in range(len(chunks) - 1):
            # Find common text between consecutive chunks
            chunk1_end = chunks[i][-100:]
            chunk2_start = chunks[i+1][:100]
            
            # Simple overlap detection
            for length in range(min(len(chunk1_end), len(chunk2_start)), 0, -1):
                if chunk1_end[-length:] == chunk2_start[:length]:
                    overlaps.append(length)
                    break
            else:
                overlaps.append(0)
        
        if overlaps:
            avg_overlap = sum(overlaps) / len(overlaps)
            print(f"  Average overlap: {avg_overlap:.0f} chars")
        else:
            print(f"  ⚠️ No overlap detected")

# Test different chunking strategies
document = """
ExaCC VM Cluster Configuration.

Network Requirements:
ExaCC VM cluster requires 3 client IP addresses and 2 backup IP addresses.
Each IP must be from the client subnet with VLAN tagging.

Backup Policy:
RMAN backup uses Level 0 retention of 7 days and incremental retention of 3 days.
Archive logs are kept for 24 hours.

Patching Guidelines:
Patching requires minimum 2-hour window.
Rolling patches available for non-CDB only.
Test patches in non-production first.
""" * 10  # Repeat for more content

# Strategy 1: Small chunks, no overlap
print("Strategy 1: Small chunks (200 chars), no overlap")
splitter1 = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
chunks1 = splitter1.split_text(document)
analyze_chunks(chunks1)

print("\n")

# Strategy 2: Medium chunks, medium overlap
print("Strategy 2: Medium chunks (500 chars), medium overlap (50)")
splitter2 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks2 = splitter2.split_text(document)
analyze_chunks(chunks2)

print("\n")

# Strategy 3: Large chunks, large overlap
print("Strategy 3: Large chunks (1000 chars), large overlap (100)")
splitter3 = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks3 = splitter3.split_text(document)
analyze_chunks(chunks3)
```

---

## Concept #5: Prompt Engineering

### 5.1 Basic RAG Prompt

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("open_api_key"))

def basic_rag_prompt(query, retrieved_chunks):
    """Simple RAG prompt template"""
    
    # Join chunks
    context = "\n\n".join(retrieved_chunks)
    
    # Build prompt
    prompt = f"""Answer the question based on the context below.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""
    
    return prompt

# Example
retrieved_chunks = [
    "ExaCC VM cluster requires 3 client IP addresses and 2 backup IP addresses.",
    "Each IP must be from the client subnet with proper VLAN tagging."
]

query = "How many IP addresses does ExaCC VM cluster need?"

prompt = basic_rag_prompt(query, retrieved_chunks)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

print(f"Query: {query}\n")
print(f"Answer: {response.choices[0].message.content}")

"""
Output:
Query: How many IP addresses does ExaCC VM cluster need?

Answer: ExaCC VM cluster requires 5 IP addresses total: 3 client IP addresses 
and 2 backup IP addresses.
"""
```

### 5.2 Strict Context-Only Prompt

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("open_api_key"))

def strict_rag_prompt(query, retrieved_chunks):
    """Prompt that prevents hallucination"""
    
    context = "\n\n".join(retrieved_chunks)
    
    prompt = f"""You must answer STRICTLY based on the context below. Do not use any external knowledge.

CONTEXT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
- If the answer IS in the context: Provide the answer
- If the answer is NOT in the context: Say "This information is not available in the provided documentation."
- Do NOT make assumptions or use external knowledge

ANSWER:"""
    
    return prompt

# Test Case 1: Answer IS in context
print("Test 1: Answer in context")
print("="*60)
chunks1 = ["ExaCC VM cluster requires 3 client IPs and 2 backup IPs."]
query1 = "How many IPs?"

prompt1 = strict_rag_prompt(query1, chunks1)
response1 = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt1}],
    temperature=0
)
print(f"Answer: {response1.choices[0].message.content}\n")

# Test Case 2: Answer NOT in context
print("Test 2: Answer NOT in context")
print("="*60)
chunks2 = ["ExaCC VM cluster requires 3 client IPs and 2 backup IPs."]
query2 = "What is the cost of ExaCC?"

prompt2 = strict_rag_prompt(query2, chunks2)
response2 = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt2}],
    temperature=0
)
print(f"Answer: {response2.choices[0].message.content}")

"""
Output:
Test 1: Answer in context
============================================================
Answer: ExaCC VM cluster requires 5 IP addresses total: 3 client IPs 
and 2 backup IPs.

Test 2: Answer NOT in context
============================================================
Answer: This information is not available in the provided documentation.

✅ LLM correctly admits when it doesn't know!
"""
```

### 5.3 Source Citation Prompt

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("open_api_key"))

def citation_rag_prompt(query, chunks_with_metadata):
    """Prompt that requires source citations"""
    
    # Format context with source numbers
    context_parts = []
    for i, item in enumerate(chunks_with_metadata, 1):
        source = item['metadata'].get('source', 'Unknown')
        page = item['metadata'].get('page', 'N/A')
        text = item['text']
        
        context_parts.append(
            f"[Source {i}] From {source} (Page {page}):\n{text}"
        )
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""Answer the question using the numbered sources below. Cite your sources.

SOURCES:
{context}

QUESTION:
{query}

INSTRUCTIONS:
- Answer using ONLY the sources above
- Cite sources as [Source 1], [Source 2], etc.
- Format: "According to [Source X], ..."
- If answer not in sources: "This information is not available."

ANSWER:"""
    
    return prompt

# Example
chunks = [
    {
        "text": "ExaCC VM cluster requires 3 client IPs and 2 backup IPs.",
        "metadata": {"source": "networking_guide.pdf", "page": 42}
    },
    {
        "text": "Each IP must be from the client subnet with VLAN tagging.",
        "metadata": {"source": "networking_guide.pdf", "page": 43}
    },
    {
        "text": "Backup retention: Level 0 kept 7 days, incremental 3 days.",
        "metadata": {"source": "backup_guide.pdf", "page": 15}
    }
]

query = "What are the IP requirements for ExaCC VM cluster?"

prompt = citation_rag_prompt(query, chunks)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

print(f"Query: {query}\n")
print(f"Answer:\n{response.choices[0].message.content}")

"""
Output:
Query: What are the IP requirements for ExaCC VM cluster?

Answer:
According to [Source 1], ExaCC VM cluster requires 3 client IPs and 2 backup IPs.
[Source 2] specifies that each IP must be from the client subnet with VLAN tagging.

✅ LLM cites specific sources!
"""
```

### 5.4 Step-by-Step Reasoning Prompt

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("open_api_key"))

def reasoning_rag_prompt(query, retrieved_chunks):
    """Prompt that encourages reasoning"""
    
    context = "\n\n".join(retrieved_chunks)
    
    prompt = f"""Answer the question using the context below. Show your reasoning.

CONTEXT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
Think step-by-step and format your response as:

RELEVANT INFORMATION:
[Extract relevant facts from context]

REASONING:
[Your analysis and calculations]

ANSWER:
[Final answer]

RESPONSE:"""
    
    return prompt

# Example
chunks = [
    "ExaCC VM cluster creation takes 45-60 minutes.",
    "ExaCC patching requires minimum 2-hour maintenance window."
]

query = "What's the total time needed for VM cluster creation and patching?"

prompt = reasoning_rag_prompt(query, chunks)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

print(f"Query: {query}\n")
print(f"Response:\n{response.choices[0].message.content}")

"""
Output:
Query: What's the total time needed for VM cluster creation and patching?

Response:
RELEVANT INFORMATION:
- VM cluster creation: 45-60 minutes
- Patching: minimum 2-hour (120 minutes) window

REASONING:
Total time = VM cluster creation time + Patching time
Minimum total = 45 minutes + 120 minutes = 165 minutes (2.75 hours)
Maximum total = 60 minutes + 120 minutes = 180 minutes (3 hours)

ANSWER:
The total time needed ranges from approximately 2.75 to 3 hours, consisting of 
45-60 minutes for VM cluster creation plus a minimum 2-hour patching window.

✅ Shows step-by-step reasoning!
"""
```

### 5.5 Confidence Scoring Prompt

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("open_api_key"))

def confidence_rag_prompt(query, retrieved_chunks):
    """Prompt that includes confidence assessment"""
    
    context = "\n\n".join(retrieved_chunks)
    
    prompt = f"""Answer the question and assess your confidence.

CONTEXT:
{context}

QUESTION:
{query}

FORMAT YOUR RESPONSE AS:
ANSWER: [your answer]

CONFIDENCE: [High/Medium/Low]

REASONING: [why this confidence level]
- High: Context explicitly contains complete answer
- Medium: Answer requires inference from context
- Low: Context only partially addresses question

RESPONSE:"""
    
    return prompt

# Test Case 1: High confidence (explicit answer)
print("Test 1: Explicit answer in context")
print("="*60)
chunks1 = ["ExaCC VM cluster requires exactly 3 client IP addresses."]
query1 = "How many client IPs for VM cluster?"

response1 = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": confidence_rag_prompt(query1, chunks1)}],
    temperature=0
)
print(response1.choices[0].message.content)

print("\n")

# Test Case 2: Medium confidence (requires inference)
print("Test 2: Requires inference")
print("="*60)
chunks2 = [
    "VM cluster has 3 nodes.",
    "Each node requires 2 IPs."
]
query2 = "Total IPs for the cluster?"

response2 = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": confidence_rag_prompt(query2, chunks2)}],
    temperature=0
)
print(response2.choices[0].message.content)

print("\n")

# Test Case 3: Low confidence (partial info)
print("Test 3: Partial information")
print("="*60)
chunks3 = ["VM cluster requires multiple IP addresses for different purposes."]
query3 = "How many IPs exactly?"

response3 = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": confidence_rag_prompt(query3, chunks3)}],
    temperature=0
)
print(response3.choices[0].message.content)

"""
This helps users understand:
- How reliable the answer is
- Whether they need to consult additional sources
- If the context was sufficient
"""
```

### 5.6 Multi-Document Synthesis Prompt

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("open_api_key"))

def synthesis_rag_prompt(query, chunks_with_metadata):
    """Prompt for synthesizing across multiple sources"""
    
    # Group by source
    sources = {}
    for chunk in chunks_with_metadata:
        source = chunk['metadata']['source']
        if source not in sources:
            sources[source] = []
        sources[source].append(chunk['text'])
    
    # Format context by source
    context_parts = []
    for source_name, texts in sources.items():
        context_parts.append(f"From {source_name}:")
        for text in texts:
            context_parts.append(f"  • {text}")
    
    context = "\n".join(context_parts)
    
    prompt = f"""Synthesize information from multiple sources to answer the question.

SOURCES:
{context}

QUESTION:
{query}

INSTRUCTIONS:
- Combine information from all sources
- Note if sources agree or conflict
- Cite which source provides each piece of information

ANSWER:"""
    
    return prompt

# Example
chunks = [
    {
        "text": "ExaCC requires 3 client IPs per VM cluster.",
        "metadata": {"source": "networking_guide_v1.pdf"}
    },
    {
        "text": "VM cluster needs 3 client and 2 backup IPs.",
        "metadata": {"source": "networking_guide_v2.pdf"}
    },
    {
        "text": "Best practice: allocate 5 IPs total for VM cluster.",
        "metadata": {"source": "best_practices.pdf"}
    }
]

query = "How many IPs should I allocate for ExaCC VM cluster?"

prompt = synthesis_rag_prompt(query, chunks)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

print(f"Query: {query}\n")
print(f"Synthesized Answer:\n{response.choices[0].message.content}")

"""
Output shows:
- Combined information from all sources
- Highlights where sources agree (3 client IPs)
- Notes additional info from different sources (backup IPs, best practices)
"""
```

---

## Complete RAG System

### Full End-to-End Implementation

```python
"""
Production-Ready RAG System
Combines all concepts: Embeddings → Chunking → Vector DB → Prompts → LLM
"""

import os
from openai import OpenAI
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional

class RAGSystem:
    """Complete RAG implementation"""
    
    def __init__(
        self,
        collection_name: str = "knowledge_base",
        db_path: str = "./rag_database",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "text-embedding-ada-002",
        llm_model: str = "gpt-4"
    ):
        """
        Initialize RAG system
        
        Args:
            collection_name: ChromaDB collection name
            db_path: Path to persist vector database
            chunk_size: Target chunk size (characters)
            chunk_overlap: Overlap between chunks
            embedding_model: OpenAI embedding model
            llm_model: OpenAI LLM model
        """
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        # Initialize clients
        api_key = os.getenv("open_api_key")
        if not api_key:
            raise ValueError("open_api_key environment variable not set")
        
        self.openai_client = OpenAI(api_key=api_key)
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=['\n\n', '\n', '. ', ' ', '']
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"✓ RAG System initialized")
        print(f"  Collection: {collection_name}")
        print(f"  Documents indexed: {self.collection.count()}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def add_documents(
        self,
        documents: List[Dict[str, any]],
        show_progress: bool = True
    ) -> int:
        """
        Add documents to RAG system
        
        Args:
            documents: List of {"text": str, "metadata": dict}
            show_progress: Print progress messages
        
        Returns:
            Number of chunks created
        """
        if show_progress:
            print(f"\nProcessing {len(documents)} documents...")
        
        # Step 1: Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc["text"])
            
            for i, chunk_text in enumerate(chunks):
                chunk_metadata = {
                    **doc.get("metadata", {}),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                all_chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
        
        if show_progress:
            print(f"  Created {len(all_chunks)} chunks")
        
        # Step 2: Generate embeddings
        if show_progress:
            print(f"  Generating embeddings...")
        
        embeddings = []
        for i, chunk in enumerate(all_chunks):
            if show_progress and (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(all_chunks)}")
            
            embedding = self._get_embedding(chunk["text"])
            embeddings.append(embedding)
        
        # Step 3: Index in ChromaDB
        if show_progress:
            print(f"  Indexing in vector database...")
        
        current_count = self.collection.count()
        
        self.collection.add(
            ids=[f"chunk_{current_count + i}" for i in range(len(all_chunks))],
            embeddings=embeddings,
            documents=[c["text"] for c in all_chunks],
            metadatas=[c["metadata"] for c in all_chunks]
        )
        
        if show_progress:
            print(f"✓ Indexed {len(all_chunks)} chunks")
            print(f"  Total documents in collection: {self.collection.count()}")
        
        return len(all_chunks)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Retrieve relevant chunks for query
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            filters: Metadata filters
        
        Returns:
            {
                "chunks": [texts],
                "metadatas": [metadata dicts],
                "distances": [similarity scores]
            }
        """
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )
        
        return {
            "chunks": results['documents'][0],
            "metadatas": results['metadatas'][0],
            "distances": results['distances'][0]
        }
    
    def _build_prompt(
        self,
        query: str,
        context_data: Dict,
        prompt_type: str = "citation"
    ) -> str:
        """
        Build RAG prompt
        
        Args:
            query: User question
            context_data: Retrieved context
            prompt_type: "basic", "strict", "citation", "reasoning"
        
        Returns:
            Formatted prompt
        """
        chunks = context_data["chunks"]
        metadatas = context_data["metadatas"]
        
        if prompt_type == "citation":
            # Format with source citations
            context_parts = []
            for i, (chunk, meta) in enumerate(zip(chunks, metadatas), 1):
                source = meta.get('source', 'Unknown')
                page = meta.get('page', 'N/A')
                context_parts.append(
                    f"[Source {i}] From {source} (Page {page}):\n{chunk}"
                )
            
            context = "\n\n".join(context_parts)
            
            prompt = f"""Answer using the numbered sources below. Cite your sources.

SOURCES:
{context}

QUESTION:
{query}

INSTRUCTIONS:
- Answer using ONLY the sources above
- Cite sources as [Source 1], [Source 2], etc.
- If answer not in sources: "This information is not available in the documentation."

ANSWER:"""
        
        elif prompt_type == "strict":
            context = "\n\n".join(chunks)
            
            prompt = f"""Answer STRICTLY from the context. Do not use external knowledge.

CONTEXT:
{context}

QUESTION:
{query}

If answer not in context, say: "Information not available in documentation."

ANSWER:"""
        
        elif prompt_type == "reasoning":
            context = "\n\n".join(chunks)
            
            prompt = f"""Answer with step-by-step reasoning.

CONTEXT:
{context}

QUESTION:
{query}

FORMAT:
RELEVANT INFO: [from context]
REASONING: [your analysis]
ANSWER: [final answer]

RESPONSE:"""
        
        else:  # basic
            context = "\n\n".join(chunks)
            prompt = f"""Answer based on context.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""
        
        return prompt
    
    def query(
        self,
        question: str,
        top_k: int = 3,
        filters: Optional[Dict] = None,
        prompt_type: str = "citation",
        temperature: float = 0,
        verbose: bool = True
    ) -> Dict:
        """
        Complete RAG query
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            filters: Metadata filters
            prompt_type: Type of prompt to use
            temperature: LLM temperature
            verbose: Print intermediate steps
        
        Returns:
            {
                "answer": str,
                "sources": [chunks],
                "metadata": [metadata dicts],
                "prompt": str (if verbose)
            }
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Query: {question}")
            print('='*60)
        
        # Retrieve context
        if verbose:
            print("\n1. Retrieving relevant context...")
        
        context_data = self.retrieve(question, top_k, filters)
        
        if verbose:
            print(f"   Retrieved {len(context_data['chunks'])} chunks:")
            for i, (chunk, dist, meta) in enumerate(zip(
                context_data['chunks'],
                context_data['distances'],
                context_data['metadatas']
            ), 1):
                similarity = 1 - dist
                source = meta.get('source', 'Unknown')
                print(f"   [{i}] {similarity:.3f} - {source}: {chunk[:60]}...")
        
        # Build prompt
        if verbose:
            print("\n2. Building prompt...")
        
        prompt = self._build_prompt(question, context_data, prompt_type)
        
        # Generate answer
        if verbose:
            print("\n3. Generating answer...")
        
        response = self.openai_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        
        answer = response.choices[0].message.content
        
        if verbose:
            print(f"\n{'='*60}")
            print("ANSWER:")
            print(answer)
            print('='*60)
        
        result = {
            "answer": answer,
            "sources": context_data["chunks"],
            "metadata": context_data["metadatas"]
        }
        
        if verbose:
            result["prompt"] = prompt
        
        return result
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.collection_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model
        }


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

if __name__ == "__main__":
    
    # Initialize RAG system
    rag = RAGSystem(
        collection_name="exacc_knowledge",
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Sample documents
    documents = [
        {
            "text": """
ExaCC VM Cluster Networking Guide

Network Requirements:
ExaCC VM cluster requires 3 client IP addresses and 2 backup IP addresses for full redundancy.
Each IP address must be allocated from the client subnet and must have proper VLAN tagging configured.

Bond Configuration:
Network bonding for ExaCC uses active-passive mode for the client network interfaces.
LACP (Link Aggregation Control Protocol) is not supported on client network interfaces.
All client IPs must have corresponding DNS entries configured before VM cluster creation.

IP Allocation Best Practices:
- Reserve IP addresses from a dedicated subnet
- Ensure IP addresses are not used by other systems
- Document IP allocation in your IPAM system
- Configure both forward and reverse DNS
            """,
            "metadata": {
                "source": "networking_guide.pdf",
                "page": 42,
                "section": "VM_Cluster",
                "version": "2024-Q1"
            }
        },
        {
            "text": """
ExaCC Backup and Recovery Guide

RMAN Configuration:
ExaCC backup configuration uses Oracle RMAN (Recovery Manager) with the following retention policy:
- Level 0 (full) backups: Retained for 7 days
- Incremental backups: Retained for 3 days  
- Archive logs: Retained for 24 hours

Backup Schedule:
The default backup schedule for ExaCC is:
- Level 0 backups run weekly on Sunday at 2:00 AM
- Incremental backups run daily at 2:00 AM
- Archive log backups run every hour on the hour

Recovery Time Objectives:
- Point-in-time recovery: Maximum 4 hours
- Full database restore: Maximum 8 hours
- Individual tablespace recovery: Maximum 2 hours
            """,
            "metadata": {
                "source": "backup_guide.pdf",
                "page": 15,
                "section": "RMAN_Config",
                "version": "2024-Q1"
            }
        },
        {
            "text": """
ExaCC Patching Guidelines

Patching Requirements:
ExaCC patching requires a minimum maintenance window of 2 hours.
Rolling patches are available only for non-CDB (non-Container Database) environments.
For production systems, always test patches in non-production environments first.

Patching Process:
1. Review patch README for prerequisites and known issues
2. Take full backup before applying patches
3. Apply patch during approved maintenance window
4. Validate database functionality after patching
5. Monitor system for 24 hours post-patch

Patch Frequency:
- Security patches: Applied monthly
- Bug fix patches: Applied quarterly  
- Feature updates: Applied semi-annually
            """,
            "metadata": {
                "source": "patching_guide.pdf",
                "page": 8,
                "section": "Patch_Process",
                "version": "2024-Q1"
            }
        }
    ]
    
    # Add documents to RAG system
    rag.add_documents(documents)
    
    # Query examples
    print("\n\n" + "="*80)
    print("QUERY EXAMPLES")
    print("="*80)
    
    # Query 1: IP requirements
    result1 = rag.query(
        "How many IP addresses are required for ExaCC VM cluster?",
        top_k=2,
        prompt_type="citation"
    )
    
    # Query 2: Backup retention
    result2 = rag.query(
        "What is the backup retention policy?",
        top_k=2,
        prompt_type="citation"
    )
    
    # Query 3: With metadata filter
    result3 = rag.query(
        "Tell me about ExaCC configuration",
        top_k=2,
        filters={"section": "VM_Cluster"},  # Only networking docs
        prompt_type="citation"
    )
    
    # Get system stats
    print("\n\n" + "="*80)
    print("SYSTEM STATISTICS")
    print("="*80)
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
```

---

## Advanced Topics

### Hybrid Search (Keyword + Semantic)

```python
import chromadb
from openai import OpenAI
import os
from rank_bm25 import BM25Okapi
import numpy as np

client_openai = OpenAI(api_key=os.getenv("open_api_key"))
client_chroma = chromadb.PersistentClient(path="./vector_db")

def hybrid_search(query, collection, top_k=5, alpha=0.5):
    """
    Combine vector search (semantic) + BM25 (keyword)
    
    Args:
        query: Search query
        collection: ChromaDB collection
        top_k: Number of results
        alpha: Weight for vector search (0-1), (1-alpha) for BM25
    
    Returns:
        Ranked results
    """
    # Get all documents
    all_docs = collection.get()
    documents = all_docs['documents']
    ids = all_docs['ids']
    
    # 1. Vector search (semantic)
    query_embedding = client_openai.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    ).data[0].embedding
    
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=len(documents)  # Get all for ranking
    )
    
    # Create vector score dict
    vector_scores = {}
    for id, dist in zip(vector_results['ids'][0], vector_results['distances'][0]):
        vector_scores[id] = 1 - dist  # Convert distance to similarity
    
    # 2. BM25 search (keyword)
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Normalize BM25 scores to 0-1
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_scores_norm = bm25_scores / max_bm25
    
    # Create BM25 score dict
    bm25_score_dict = {id: score for id, score in zip(ids, bm25_scores_norm)}
    
    # 3. Combine scores
    combined_scores = {}
    for id in ids:
        vector_score = vector_scores.get(id, 0)
        bm25_score = bm25_score_dict.get(id, 0)
        
        # Weighted combination
        combined_scores[id] = alpha * vector_score + (1 - alpha) * bm25_score
    
    # 4. Rank and return top-k
    ranked_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:top_k]
    
    results = collection.get(
        ids=ranked_ids,
        include=["documents", "metadatas"]
    )
    
    return results, combined_scores

# Example
collection = client_chroma.get_collection("exacc_docs")

query = "IP address configuration"

results, scores = hybrid_search(query, collection, top_k=3, alpha=0.7)

print(f"Hybrid Search Results for: '{query}'\n")
for id, doc in zip(results['ids'], results['documents']):
    print(f"Score: {scores[id]:.3f}")
    print(f"Document: {doc[:80]}...")
    print()
```

### Query Rewriting

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("open_api_key"))

def rewrite_query(original_query):
    """
    Rewrite vague query to be more specific for better retrieval
    """
    prompt = f"""Rewrite this search query to be more specific and complete.
Add relevant keywords that would help find the answer.

Original query: {original_query}

Rewritten query:"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    return response.choices[0].message.content

# Examples
queries = [
    "IPs for VM?",
    "backup time?",
    "patch?"
]

for query in queries:
    rewritten = rewrite_query(query)
    print(f"Original:  {query}")
    print(f"Rewritten: {rewritten}")
    print()

"""
Output:
Original:  IPs for VM?
Rewritten: How many IP addresses are required for ExaCC VM cluster configuration?

Original:  backup time?
Rewritten: What is the backup schedule and timing for ExaCC database backups?

Original:  patch?
Rewritten: What are the ExaCC patching requirements, process, and maintenance window?

Better queries → Better retrieval → Better answers
"""
```

### Re-ranking Retrieved Results

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("open_api_key"))

def rerank_chunks(query, chunks, top_k=3):
    """
    Re-rank retrieved chunks using cross-encoder (more accurate but slower)
    
    In production, use dedicated re-ranking models like:
    - Cohere Rerank API
    - sentence-transformers cross-encoders
    
    This example uses LLM for simplicity
    """
    # Score each chunk
    scores = []
    
    for chunk in chunks:
        prompt = f"""Rate how relevant this text is to the query on a scale of 0-10.

Query: {query}

Text: {chunk}

Respond with ONLY a number between 0-10.
Score:"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
        except:
            score = 0
        
        scores.append(score)
    
    # Rank by score
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    # Return top-k
    return [chunks[i] for i in ranked_indices[:top_k]]

# Example
query = "How many IP addresses for VM cluster?"

# Initial retrieval (might include some less relevant chunks)
chunks = [
    "ExaCC VM cluster requires 3 client IPs and 2 backup IPs.",
    "Network bonding uses active-passive mode.",
    "DNS entries required before cluster creation.",
    "VLAN tagging is mandatory for all IPs.",
    "Backup retention is 7 days for Level 0."
]

# Re-rank
reranked = rerank_chunks(query, chunks, top_k=3)

print(f"Query: {query}\n")
print("Top 3 after re-ranking:")
for i, chunk in enumerate(reranked, 1):
    print(f"{i}. {chunk}")

"""
Re-ranking improves precision by:
- Using more sophisticated relevance scoring
- Considering query-document interaction
- Filtering out less relevant results
"""
```

---

## Production Deployment

### Performance Optimization

```python
# 1. Batch Embedding
def embed_batch_efficient(texts, batch_size=100):
    """Process embeddings in batches"""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=batch
        )
        
        embeddings.extend([item.embedding for item in response.data])
    
    return embeddings

# 2. Connection Pooling
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("open_api_key"),
    max_retries=3,
    timeout=30.0
)

# 3. Caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embedding(text):
    """Cache embeddings for frequently queried texts"""
    return get_embedding(text)

# 4. Async Processing
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI(api_key=os.getenv("open_api_key"))

async def async_embed(texts):
    """Async embedding for better throughput"""
    tasks = [
        async_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        for text in texts
    ]
    
    responses = await asyncio.gather(*tasks)
    return [r.data[0].embedding for r in responses]
```

### Error Handling

```python
from openai import OpenAI, APIError, RateLimitError
import time

client = OpenAI(api_key=os.getenv("open_api_key"))

def robust_embedding(text, max_retries=3):
    """Embedding with retry logic"""
    
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        
        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
        
        except APIError as e:
            print(f"API error: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                raise
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise
    
    raise Exception("Max retries exceeded")
```

### Monitoring

```python
import time
from datetime import datetime

class RAGMonitor:
    """Track RAG system metrics"""
    
    def __init__(self):
        self.queries = []
        self.latencies = []
        self.errors = []
    
    def log_query(self, query, latency, success, error=None):
        """Log query metrics"""
        self.queries.append({
            "timestamp": datetime.now(),
            "query": query,
            "latency_ms": latency,
            "success": success,
            "error": str(error) if error else None
        })
        
        self.latencies.append(latency)
        
        if not success:
            self.errors.append(error)
    
    def get_stats(self):
        """Get system statistics"""
        if not self.latencies:
            return {}
        
        return {
            "total_queries": len(self.queries),
            "avg_latency_ms": sum(self.latencies) / len(self.latencies),
            "p50_latency_ms": sorted(self.latencies)[len(self.latencies)//2],
            "p95_latency_ms": sorted(self.latencies)[int(len(self.latencies)*0.95)],
            "error_rate": len(self.errors) / len(self.queries) if self.queries else 0,
            "total_errors": len(self.errors)
        }

# Usage
monitor = RAGMonitor()

def monitored_query(rag_system, query):
    """Query with monitoring"""
    start = time.time()
    
    try:
        result = rag_system.query(query, verbose=False)
        latency = (time.time() - start) * 1000
        monitor.log_query(query, latency, True)
        return result
    
    except Exception as e:
        latency = (time.time() - start) * 1000
        monitor.log_query(query, latency, False, e)
        raise

# Print stats periodically
print(monitor.get_stats())
```

---

**Last Updated:** January 2025
