import gzip, json
import re, html
import time

import torch
import numpy as np
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")


def clean_text(text: str) -> str:
    """
    Cleans review text for embedding conversion by removing noise
    while preserving semantic meaning.

    Steps:
    - Unescape HTML entities
    - Lowercase the text
    - Remove URLs, mentions, hashtags
    - Remove HTML tags
    - Remove non-alphanumeric characters (except basic punctuation)
    - Normalize whitespace
    """

    if not isinstance(text, str):
        return ""

    # 1. Unescape HTML entities (e.g. &amp; → &)
    text = html.unescape(text)

    # 2. Lowercase
    text = text.lower()

    # 3. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 4. Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # 5. Remove @mentions and #hashtags
    text = re.sub(r"[@#]\w+", "", text)

    # 6. Remove non-alphanumeric chars (but keep .,!?;:'- for context)
    text = re.sub(r"[^a-z0-9\s\.,!?;:'\"-]", "", text)

    # 7. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


reviews = dict()
file = 'data/Appliances.jsonl.gz'
with gzip.open(file, 'rt', encoding='utf-8') as f:
    for line in f:
        record = json.loads(line)
        text = record['title'] + ' ' + record['text']
        text = clean_text(text)
        if len(text.split(' ')) > 5:
            reviews.setdefault(record['parent_asin'], []).append(text)

start_time = time.time()
total = len(reviews.keys())
idx = 1

for key in reviews.keys():
    if idx % 100 == 0:
        elapsed = time.time() - start_time
        print(f'{idx} / {total} - Elapsed time: {elapsed:.2f} seconds')

    embeddings = model.encode(
        reviews[key], 
        batch_size=700,
        convert_to_numpy=False, 
        normalize_embeddings=True
    )
    product_embedding = embeddings.mean(dim=0)
    reviews[key] = product_embedding.cpu().numpy().tolist()

    idx += 1

with gzip.open("embed_Appliances.jsonl.gz", "wt") as f:
    for key, emb in reviews.items():
        f.write(json.dumps({"parent_asin": key, "embed": emb}) + "\n")