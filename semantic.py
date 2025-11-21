import gzip, json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Embedding Retrieval ----------
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

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

query = 'a small mini fridge for storing drinks in college'
query_embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

# Stack embeddings into a matrix
product_ids = list(reviews.keys())
emb_matrix = np.stack([reviews[pid] for pid in product_ids])

# Compute cosine similarity with the query
similarities = cosine_similarity([query_embedding], emb_matrix)[0]

# Rank products by similarity
ranked = sorted(zip(product_ids, similarities), key=lambda x: x[1], reverse=True)

top_pid = ranked[0][0]
product_meta = products[top_pid]

print(f'Top recommended product: {products[top_pid]['title']}')
print(f'ASIN: {top_pid}')

# ---------- Retrieval Augmented Generation ----------

from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.chains import RetrievalQA
import json

MODEL_NAME = "llama3"

# Use SentenceTransformer embeddings as before
class STEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

emb = STEmbeddings(model)

# Create vectorstore
product_text = json.dumps(product_meta, indent=2)
store = Chroma.from_texts([product_text], embedding=emb)
retriever = store.as_retriever()

# Load from Ollama
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.2,
    num_ctx=4096,
)

# QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
)

user_question = 'Is this fridge quiet enough for college dorm use?'
answer = qa.invoke({'query': user_question})

print('\nRAG Answer:')
print(answer['result'])