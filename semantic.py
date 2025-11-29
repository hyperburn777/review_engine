import gzip, json
import requests
import torch, re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoProcessor, AutoModelForImageTextToText
from io import BytesIO
from summarize import Summarizer


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

query = input("Please give a short description on what you are looking for:\n")
query_embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

# Stack embeddings into a matrix
product_ids = list(reviews.keys())
emb_matrix = np.stack([reviews[pid] for pid in product_ids])

# Compute cosine similarity with the query
similarities = cosine_similarity([query_embedding], emb_matrix)[0]

# Rank products by similarity
ranked = sorted(zip(product_ids, similarities), key=lambda x: x[1], reverse=True)

summarizer = Summarizer(device='cuda')

print("Top Recommended Items:\n")

for idx, (pid, score) in enumerate(ranked[:10]):
    summary = summarizer.generate_summary(products[pid]['title'], products[pid]['main_category'], products[pid]['average_rating'], products[pid]['images'])
    print(f'Rank: {idx+1}, Name: {products[pid]['title']}, Score: {score}, Summary: {summary}\n')