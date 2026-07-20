#!/usr/bin/env python3
# Copyright 2026 SingleStore, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple before/after demonstration of cross-encoder reranking.
Shows top 5 results before and after reranking side-by-side.
"""
import pymysql
import json
import sys
import time
from sentence_transformers import SentenceTransformer, CrossEncoder

with open('config.json') as f:
    config = json.load(f)

class BeforeAfterDemo:
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
            print("Loading bi-encoder (one-time)...")
            self._bi_encoder = SentenceTransformer('BAAI/bge-large-en-v1.5')
        return self._bi_encoder

    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            print("Loading cross-encoder (one-time)...")
            self._cross_encoder = CrossEncoder('BAAI/bge-reranker-v2-m3')
        return self._cross_encoder

    def get_candidates(self, query, num_candidates=50):
        """Retrieve candidates using vector search."""
        emb = self.bi_encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        emb_json = json.dumps(emb.tolist())

        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT chunk_id, text, DOT_PRODUCT(embedding_1024, %s :> VECTOR(1024)) as score
                FROM {config['singlestore']['table_name']}
                ORDER BY score DESC LIMIT %s
            """, (emb_json, num_candidates))
            return cur.fetchall()

    def rerank(self, query, candidates):
        """Apply cross-encoder reranking."""
        pairs = [[query, text] for _, text, _ in candidates]
        scores = self.cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)

        reranked = [(candidates[i][0], candidates[i][1], float(scores[i]))
                    for i in range(len(candidates))]
        reranked.sort(key=lambda x: x[2], reverse=True)
        return reranked

    def compare(self, query):
        print("=" * 80)
        print(f"Query: '{query}'")
        print("=" * 80 + "\n")

        # Get 50 candidates
        print("Retrieving 50 candidates from SingleStore...")
        start = time.time()
        candidates = self.get_candidates(query, num_candidates=50)
        retrieval_time = time.time() - start
        print(f"Retrieved in {retrieval_time:.3f}s\n")

        # Apply cross-encoder reranking
        print("Applying cross-encoder reranking...")
        start = time.time()
        reranked = self.rerank(query, candidates)
        rerank_time = time.time() - start
        print(f"Reranked in {rerank_time:.3f}s\n")

        # Show before/after top 5
        before = candidates[:5]
        after = reranked[:5]

        print("=" * 80)
        print("BEFORE RERANKING (Vector Search Top 5)")
        print("=" * 80 + "\n")

        for i, (chunk_id, text, score) in enumerate(before, 1):
            print(f"{i}. Chunk {chunk_id} | Vector Score: {score:.4f}")
            print(f"   {text[:200].strip()}...")
            print()

        print("=" * 80)
        print("AFTER RERANKING (Cross-Encoder Top 5)")
        print("=" * 80 + "\n")

        for i, (chunk_id, text, score) in enumerate(after, 1):
            print(f"{i}. Chunk {chunk_id} | Rerank Score: {score:.4f}")
            print(f"   {text[:200].strip()}...")
            print()

        # Show what changed at rank 1
        print("=" * 80)
        print("RANK 1 COMPARISON")
        print("=" * 80 + "\n")

        before_id = before[0][0]
        after_id = after[0][0]

        if before_id == after_id:
            print("Rank 1 unchanged:")
            print(f"  Chunk {before_id} remained at position 1")
            print(f"  Vector score: {before[0][2]:.4f}")
            print(f"  Rerank score: {after[0][2]:.4f}")
        else:
            # Find where the new rank 1 was before
            old_position = None
            for i, (cid, _, _) in enumerate(before, 1):
                if cid == after_id:
                    old_position = i
                    break

            print("Rank 1 changed:")
            print(f"  Before: Chunk {before_id} (vector score {before[0][2]:.4f})")
            print(f"  After:  Chunk {after_id} (rerank score {after[0][2]:.4f})")
            if old_position:
                print(f"  Movement: Position {old_position} → Position 1")
            else:
                print(f"  Movement: Outside top 5 → Position 1")

        print("\n" + "=" * 80)
        print("TIMING")
        print("=" * 80)
        print(f"Retrieval: {retrieval_time:.3f}s")
        print(f"Reranking: {rerank_time:.3f}s")
        print(f"Total:     {retrieval_time + rerank_time:.3f}s")
        print("=" * 80 + "\n")

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rerank_example.py 'your query'")
        print('Example: python rerank_example.py "puzzle game with falling blocks"')
        sys.exit(1)

    demo = BeforeAfterDemo()
    try:
        demo.compare(sys.argv[1])
    finally:
        demo.close()
