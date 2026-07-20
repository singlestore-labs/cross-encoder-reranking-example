-- Copyright 2026 SingleStore, Inc.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
-- http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

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
