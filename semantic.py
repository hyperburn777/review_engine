import gzip, json
import requests
import torch, re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoProcessor, AutoModelForImageTextToText
from io import BytesIO
from summarize import Summarizer
from rag import RAG
from rank import rank

PRODUCTS = 3

model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

products = dict()
file = "data/meta_Appliances.jsonl.gz"
with gzip.open(file, "rt", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        products[record["parent_asin"]] = {
            k: v for k, v in record.items() if k != "parent_asin"
        }

reviews = dict()
file = "data/embed_Appliances.jsonl.gz"
with gzip.open(file, "rt", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        reviews[record["parent_asin"]] = record["embed"]

query = input("Please give a short description on what you are looking for:\n")
query_embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

# Stack embeddings into a matrix
product_ids = list(reviews.keys())
emb_matrix = np.stack([reviews[pid] for pid in product_ids])

# Compute cosine similarity with the query
similarities = cosine_similarity([query_embedding], emb_matrix)[0]

# Rank products by using multi-factor + mmr
ranked = rank(
    product_ids,
    similarities,
    products,
    reviews,
    query,
    top_k_candidates=200,
    final_k=PRODUCTS
)

summarizer = Summarizer(device="cuda")

print("Top Recommended Items:\n")

for idx, (pid, score) in enumerate(ranked[:PRODUCTS]):
    summary = summarizer.generate_summary(
        products[pid]["title"],
        products[pid]["main_category"],
        products[pid]["average_rating"],
        products[pid]["images"],
    )
    print(
        f"Rank: {idx+1}, Name: {products[pid]['title']}, Score: {score}, Summary: {summary}\n"
    )

rag = RAG(model, default_meta=products[ranked[0][0]])

print("If you have any questions about the products, please ask.\n")

print(f"Now focusing on the product at rank 1, which is {products[ranked[0][0]]['title']}, if you want to change, please follow this format: rank #.\n")

print("Or if you don't have any questions, please type 'exit' to end this procedure.\n")
while True:
    query = input("Please input your questions or instructions:\n")
    if query == 'exit':
        break
    else:
        strings = query.strip().split(" ")

        if len(strings) == 2 and strings[0] == 'rank':

            if strings[1].isdigit():

                num = int(strings[1])

                if num > PRODUCTS:

                    print(f"Please choose a number between 1 and {PRODUCTS}.\n")

                else:

                    product_meta = products[ranked[num][0]]

                    rag.change_product(product_meta)

                    print(f"Now focusing on the product at rank {num}, which is {products[ranked[num][0]]['title']}.\n")
            else:

                print("please input a number.\n")
        else:

            print("RAG Answer:\n")
            print(rag.generate_answer(query) + '\n')