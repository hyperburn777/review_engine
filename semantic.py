import gzip, json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

products = dict()
file = 'data/meta_Appliances.jsonl.gz'
with gzip.open(file, 'rt', encoding='utf-8') as f:
    for line in f:
        record = json.loads(line)
        products[record['parent_asin']] = {k: v for k, v in record.items() if k != 'parent_asin'}

reviews = dict()
file = 'data/embed_Appliances.jsonl.gz'
with gzip.open(file, 'rt', encoding='utf-8') as f:
    for line in f:
        record = json.loads(line)
        reviews[record['parent_asin']] = record['embed']

query = "a small mini fridge for storing drinks in college"
query_embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

# Stack embeddings into a matrix
product_ids = list(reviews.keys())
emb_matrix = np.stack([reviews[pid] for pid in product_ids])

# Compute cosine similarity with the query
similarities = cosine_similarity([query_embedding], emb_matrix)[0]

# Rank products by similarity
ranked = sorted(zip(product_ids, similarities), key=lambda x: x[1], reverse=True)

for pid, score in ranked[:10]:
    print(products[pid]['title'], score)