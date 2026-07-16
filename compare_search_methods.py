#!/usr/bin/env python3
"""
Three-way comparison:
1. Vector Search
2. Hybrid Search with RRF
3. Hybrid Search with RRF + Cross-Encoder Reranking
"""
import pymysql
import json
import sys
import time
from sentence_transformers import SentenceTransformer, CrossEncoder

with open('config.json') as f:
    config = json.load(f)

class SearchComparison:
    def __init__(self):
        self._bi_encoder = None
        self._cross_encoder = None
        self.conn = pymysql.connect(
            host=config['singlestore']['host'],
            port=config['singlestore']['port'],
            user=config['singlestore']['user'],
            password=config['singlestore'].get('password', ''),
            database=config['singlestore']['database']
        )
        print("Connected to SingleStore\n")

    @property
    def bi_encoder(self):
        if self._bi_encoder is None:
            print("Loading bi-encoder (one-time, ~10s)...")
            self._bi_encoder = SentenceTransformer('BAAI/bge-large-en-v1.5')
        return self._bi_encoder

    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            print("Loading cross-encoder (one-time, ~10s)...")
            self._cross_encoder = CrossEncoder('BAAI/bge-reranker-v2-m3')
        return self._cross_encoder

    def method1_vector(self, query):
        """Method 1: Vector Search Only"""
        print("METHOD 1: Vector Search")
        start = time.time()

        emb = self.bi_encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        emb_json = json.dumps(emb.tolist())

        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT chunk_id, text, DOT_PRODUCT(embedding_1024, %s :> VECTOR(1024)) as score
                FROM {config['singlestore']['table_name']}
                ORDER BY score DESC LIMIT 5
            """, (emb_json,))
            results = cur.fetchall()

        elapsed = time.time() - start
        print(f"  Time: {elapsed:.3f}s\n")
        return results, elapsed

    def method2_hybrid_rrf(self, query):
        """Method 2: Hybrid Search with RRF"""
        print("METHOD 2: Hybrid + RRF")
        start = time.time()

        emb = self.bi_encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        emb_json = json.dumps(emb.tolist())

        with self.conn.cursor() as cur:
            cur.execute(f"""
                WITH vector_results AS (
                    SELECT chunk_id, text,
                           DOT_PRODUCT(embedding_1024, %s :> VECTOR(1024)) as vec_score,
                           ROW_NUMBER() OVER (ORDER BY DOT_PRODUCT(embedding_1024, %s :> VECTOR(1024)) DESC) as vec_rank
                    FROM {config['singlestore']['table_name']}
                    ORDER BY vec_score DESC LIMIT 50
                ),
                text_results AS (
                    SELECT chunk_id, text,
                           MATCH(text) AGAINST(%s) as text_score,
                           ROW_NUMBER() OVER (ORDER BY MATCH(text) AGAINST(%s) DESC) as text_rank
                    FROM {config['singlestore']['table_name']}
                    WHERE MATCH(text) AGAINST(%s)
                    LIMIT 50
                ),
                combined AS (
                    SELECT
                        COALESCE(v.chunk_id, t.chunk_id) as chunk_id,
                        COALESCE(v.text, t.text) as text,
                        (1.0 / (60 + COALESCE(v.vec_rank, 999))) +
                        (1.0 / (60 + COALESCE(t.text_rank, 999))) as rrf_score
                    FROM vector_results v
                    FULL OUTER JOIN text_results t USING (chunk_id)
                )
                SELECT chunk_id, text, rrf_score FROM combined
                ORDER BY rrf_score DESC LIMIT 5
            """, (emb_json, emb_json, query, query, query))
            results = cur.fetchall()

        elapsed = time.time() - start
        print(f"  Time: {elapsed:.3f}s\n")
        return results, elapsed

    def method3_hybrid_rerank(self, query):
        """Method 3: Hybrid + RRF + Cross-Encoder"""
        print("METHOD 3: Hybrid + RRF + Cross-Encoder")
        start = time.time()

        emb = self.bi_encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        emb_json = json.dumps(emb.tolist())

        with self.conn.cursor() as cur:
            cur.execute(f"""
                WITH vector_results AS (
                    SELECT chunk_id, text,
                           DOT_PRODUCT(embedding_1024, %s :> VECTOR(1024)) as vec_score,
                           ROW_NUMBER() OVER (ORDER BY DOT_PRODUCT(embedding_1024, %s :> VECTOR(1024)) DESC) as vec_rank
                    FROM {config['singlestore']['table_name']}
                    ORDER BY vec_score DESC LIMIT 50
                ),
                text_results AS (
                    SELECT chunk_id, text,
                           MATCH(text) AGAINST(%s) as text_score,
                           ROW_NUMBER() OVER (ORDER BY MATCH(text) AGAINST(%s) DESC) as text_rank
                    FROM {config['singlestore']['table_name']}
                    WHERE MATCH(text) AGAINST(%s)
                    LIMIT 50
                ),
                combined AS (
                    SELECT
                        COALESCE(v.chunk_id, t.chunk_id) as chunk_id,
                        COALESCE(v.text, t.text) as text,
                        (1.0 / (60 + COALESCE(v.vec_rank, 999))) +
                        (1.0 / (60 + COALESCE(t.text_rank, 999))) as rrf_score
                    FROM vector_results v
                    FULL OUTER JOIN text_results t USING (chunk_id)
                )
                SELECT chunk_id, text FROM combined
                ORDER BY rrf_score DESC LIMIT 50
            """, (emb_json, emb_json, query, query, query))
            candidates = cur.fetchall()

        retrieval_time = time.time() - start

        start = time.time()
        pairs = [[query, text] for _, text in candidates]
        scores = self.cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)

        reranked = [(candidates[i][0], candidates[i][1], float(scores[i]))
                    for i in range(len(candidates))]
        reranked.sort(key=lambda x: x[2], reverse=True)

        rerank_time = time.time() - start
        total = retrieval_time + rerank_time

        print(f"  Retrieval: {retrieval_time:.3f}s")
        print(f"  Reranking: {rerank_time:.3f}s")
        print(f"  Total: {total:.3f}s\n")

        return reranked[:5], total

    def compare(self, query):
        print("=" * 80)
        print(f"Query: '{query}'")
        print("=" * 80 + "\n")

        r1, t1 = self.method1_vector(query)
        r2, t2 = self.method2_hybrid_rrf(query)
        r3, t3 = self.method3_hybrid_rerank(query)

        print("=" * 80)
        print("TIMING SUMMARY")
        print("=" * 80)
        print(f"Method 1 (Vector):      {t1:.3f}s")
        print(f"Method 2 (Hybrid+RRF):  {t2:.3f}s")
        print(f"Method 3 (+ Rerank):    {t3:.3f}s")
        print("=" * 80 + "\n")

        for i in range(5):
            print(f"RANK {i+1}")
            print("─" * 80)

            print(f"M1 (Vector): Chunk {r1[i][0]} | Score {r1[i][2]:.4f}")
            print(f"   {r1[i][1][:120]}...\n")

            print(f"M2 (Hybrid): Chunk {r2[i][0]} | Score {r2[i][2]:.4f}")
            print(f"   {r2[i][1][:120]}...\n")

            print(f"M3 (Rerank): Chunk {r3[i][0]} | Score {r3[i][2]:.4f}")
            print(f"   {r3[i][1][:120]}...\n")

        print("=" * 80)

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_search_methods.py 'your query'")
        print('Example: python compare_search_methods.py "racing games with power-ups"')
        sys.exit(1)

    comp = SearchComparison()
    try:
        comp.compare(sys.argv[1])
    finally:
        comp.close()
