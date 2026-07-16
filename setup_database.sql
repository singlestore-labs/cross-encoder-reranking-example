-- Create database for cross-encoder reranking example
CREATE DATABASE IF NOT EXISTS video_games_search;
USE video_games_search;

-- Create table with text, embeddings, and indexes
CREATE TABLE IF NOT EXISTS documents (
    chunk_id INT AUTO_INCREMENT PRIMARY KEY,
    text TEXT,
    embedding_1024 VECTOR(1024),
    FULLTEXT(text),
    VECTOR INDEX ivf_emb(embedding_1024)
        INDEX_OPTIONS '{"index_type":"IVF_PQFS"}'
);
