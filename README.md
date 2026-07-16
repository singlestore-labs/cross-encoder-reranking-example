# Cross-Encoder Reranking Example

Demonstrates how to improve search quality by reranking SingleStore vector search results with a cross-encoder model. Takes 50 results from SingleStore vector search and uses a cross-encoder to rerank them, producing a higher quality top 5. Shows before/after comparison to prove the improvement.

**Pipeline:**
1. Retrieve 50 candidates from SingleStore using vector search
2. Score each candidate with cross-encoder model (query + document together)
3. Rerank by cross-encoder scores
4. Return top 5 results

## Requirements

- SingleStore database
- Python 3.8+
- 4GB+ memory

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Installs: `pymysql`, `sentence-transformers`, `torch`, `tqdm`, `wikipedia-api`

### 2. Create Database=

```sql
CREATE DATABASE video_games_search;
USE video_games_search;

CREATE TABLE documents (
    chunk_id INT AUTO_INCREMENT PRIMARY KEY,
    text TEXT,
    embedding_1024 VECTOR(1024),
    FULLTEXT(text),
    VECTOR INDEX ivf_emb(embedding_1024)
        INDEX_OPTIONS '{"index_type":"IVF_PQFS"}'
);
```

### 3. Configure Connection

Copy the example config and edit with your database details:

```bash
cp config.json.example config.json
```

Edit `config.json`:
```json
{
  "singlestore": {
    "host": "127.0.0.1",
    "port": 3307,
    "user": "root",
    "password": "",
    "database": "video_games_search",
    "table_name": "documents"
  }
}
```

### 4. Load Data

Downloads Wikipedia articles, chunks them, loads into database, and generates embeddings:

```bash
python setup_data.py
```

Takes 5-10 minutes. Downloads 30 Wikipedia video game articles and creates ~600 text chunks with embeddings.

## Usage

### Basic Example

```bash
python rerank_example.py "puzzle game with falling blocks"
```

Output shows:
- Top 5 before reranking
- Top 5 after reranking
- What changed at rank 1
- Timing breakdown

### Compare Methods

```bash
python compare_search_methods.py "your query"
```

Shows three-way comparison:
- Vector search only
- Hybrid search with RRF
- Hybrid search with cross-encoder reranking

### Try These Queries

```bash
python rerank_example.py "puzzle game with falling blocks"
python rerank_example.py "first person shooter"
python rerank_example.py "survival game with crafting"
python rerank_example.py "open world fantasy RPG"
```

## Files

- `setup_data.py` - Complete data setup (download + chunk + load + embed)
- `rerank_example.py` - Simple before/after comparison
- `compare_search_methods.py` - Three-way comparison
- `setup_database.sql` - Database schema
- `config.json.example` - Connection settings template
- `requirements.txt` - Python dependencies

## Models Used

- **Bi-encoder:** BAAI/bge-large-en-v1.5 (500MB, for embeddings)
- **Cross-encoder:** BAAI/bge-reranker-v2-m3 (2.3GB, for reranking)

Models download automatically on first run.

