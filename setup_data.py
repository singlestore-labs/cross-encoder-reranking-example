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
Complete data setup: downloads Wikipedia articles, chunks text, loads into SingleStore, and generates embeddings.
"""
import wikipediaapi
import time
import os
import json
import pymysql
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CONFIG_FILE = 'config.json'
EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'
EMBEDDING_DIMENSION = 1024
BATCH_SIZE = 10

GAMES = [
    "Mario Kart", "Super Mario Kart", "Mario Kart 64", "Mario Kart 8",
    "Super Mario Bros.", "The Legend of Zelda", "Pokémon Red and Blue",
    "Minecraft", "Fortnite", "Grand Theft Auto V", "The Last of Us",
    "God of War (2018 video game)", "Red Dead Redemption 2",
    "The Witcher 3: Wild Hunt", "Dark Souls", "Street Fighter II",
    "Sonic the Hedgehog", "Pac-Man", "Tetris", "Space Invaders",
    "Donkey Kong", "The Elder Scrolls V: Skyrim", "Mass Effect 2",
    "Portal (video game)", "Half-Life 2", "BioShock",
    "Halo: Combat Evolved", "Call of Duty 4: Modern Warfare",
    "Final Fantasy VII", "Chrono Trigger",
]

with open(CONFIG_FILE) as f:
    config = json.load(f)

def download_articles():
    """Download Wikipedia articles."""
    wiki = wikipediaapi.Wikipedia('CrossEncoderDemo/1.0', 'en')
    os.makedirs('data', exist_ok=True)
    output = []

    print(f"Step 1: Downloading {len(GAMES)} Wikipedia articles...")
    for i, game in enumerate(GAMES, 1):
        print(f"  [{i}/{len(GAMES)}] {game}")
        page = wiki.page(game)
        if page.exists():
            output.append(f"=== {page.title} ===\n\n")
            output.append(page.text)
            output.append("\n\n")
            time.sleep(0.5)

    with open('data/video_games.txt', 'w', encoding='utf-8') as f:
        f.writelines(output)

    text = ''.join(output)
    print(f"  Downloaded {len(GAMES)} articles ({len(text):,} characters)\n")
    return 'data/video_games.txt'

def chunk_text(text_file, chunk_size=500):
    """Chunk text into pieces."""
    print(f"Step 2: Chunking text...")
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    print(f"  Created {len(chunks)} chunks\n")
    return chunks

def load_chunks_to_db(chunks):
    """Load chunks into SingleStore."""
    print(f"Step 3: Loading chunks into SingleStore...")
    conn = pymysql.connect(
        host=config['singlestore']['host'],
        port=config['singlestore']['port'],
        user=config['singlestore']['user'],
        password=config['singlestore'].get('password', ''),
        database=config['singlestore']['database']
    )

    table_name = config['singlestore']['table_name']
    cursor = conn.cursor()
    for chunk in tqdm(chunks, desc="  Inserting"):
        cursor.execute(f"INSERT INTO {table_name} (text) VALUES (%s)", (chunk,))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"  Loaded {len(chunks)} chunks\n")

def generate_embeddings():
    """Generate embeddings in batches."""
    print(f"Step 4: Generating embeddings (this takes 5-10 minutes)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    conn = pymysql.connect(
        host=config['singlestore']['host'],
        port=config['singlestore']['port'],
        user=config['singlestore']['user'],
        password=config['singlestore'].get('password', ''),
        database=config['singlestore']['database']
    )

    table_name = config['singlestore']['table_name']
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT COUNT(*) FROM {table_name}
        WHERE embedding_{EMBEDDING_DIMENSION} IS NULL
    """)
    total_missing = cursor.fetchone()[0]

    processed = 0
    with tqdm(total=total_missing, desc="  Generating") as pbar:
        while True:
            cursor.execute(f"""
                SELECT chunk_id, text FROM {table_name}
                WHERE embedding_{EMBEDDING_DIMENSION} IS NULL
                LIMIT {BATCH_SIZE}
            """)
            rows = cursor.fetchall()
            if not rows:
                break

            chunk_ids = [row[0] for row in rows]
            texts = [row[1] for row in rows]
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

            for i, chunk_id in enumerate(chunk_ids):
                embedding_json = json.dumps(embeddings[i].tolist())
                cursor.execute(f"""
                    UPDATE {table_name}
                    SET embedding_{EMBEDDING_DIMENSION} = %s :> VECTOR({EMBEDDING_DIMENSION})
                    WHERE chunk_id = %s
                """, (embedding_json, chunk_id))

            conn.commit()
            processed += len(rows)
            pbar.update(len(rows))

    cursor.close()
    conn.close()
    print(f"  Generated {processed} embeddings\n")

def main():
    print("=" * 60)
    print("DATA SETUP")
    print("=" * 60 + "\n")

    text_file = download_articles()
    chunks = chunk_text(text_file)
    load_chunks_to_db(chunks)
    generate_embeddings()

    print("=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nYou can now run:")
    print('  python rerank_example.py "puzzle game with falling blocks"')

if __name__ == "__main__":
    main()
