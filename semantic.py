import gzip, json
import requests
import torch, re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoProcessor, AutoModelForImageTextToText
from io import BytesIO
from PIL import Image


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

summary_model_id = "llava-hf/llava-1.5-7b-hf"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = AutoProcessor.from_pretrained(summary_model_id)
summary_model = AutoModelForImageTextToText.from_pretrained(summary_model_id, dtype=dtype, device_map='auto')

def load_images(records, max_images=3):
    images = []
    for record in records[:max_images]:
        url = record['large']
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        images.append(image)
    return images

def build_messages(name, category, rating, images, tone):
    content = []

    for _ in images:
        content.append({"type": "image"})
    
    content.append({
        "type": "text",
        "text": (
            "You are a product copywriter. Write ONE concise customer-facing blurb (35–45 words). "
            "Use only what is clearly seen plus the metadata. Include the rating as X.X/5. "
            "Avoid brand/spec guesses or hype.\n\n"
            f"Product name: {name}\n"
            f"Category: {category}\n"
            f"Average rating: {rating:.1f}/5\n"
            f"Tone: {tone}"
        )
    })

    return [{"role": "user", "content": content}]


def summarize_product(name, category, rating, image_records, tone="concise"):
    images = load_images(image_records, 1)

    message = build_messages(name, category, rating, images, tone)

    text = processor.apply_chat_template(message, add_generation_prompt=True)

    inputs = processor(images=images, text=text, return_tensors="pt").to(device)

    out = summary_model.generate(**inputs, max_new_tokens=64, temperature=0.4, do_sample=False)

    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()

def extract_summary(raw: str) -> str:
    # if the whole thing is quoted, drop outer quotes
    raw = raw.strip().strip('"').strip("'")

    # grab everything after the last "ASSISTANT:"
    m = re.search(r'ASSISTANT:\s*(.*)\s*$', raw, flags=re.S)
    if m:
        summary = m.group(1).strip()
    else:
        # fallback: no label found → return the whole thing trimmed
        summary = raw.strip()

    # clean common trailing artifacts
    summary = re.sub(r'(</s>|\[END\]|\[/ASSISTANT\])\s*$', '', summary).strip()
    return summary

for pid, score in ranked[:10]:
    summary = extract_summary(summarize_product(products[pid]['title'], products[pid]['main_category'], products[pid]['average_rating'], products[pid]['images']))
    print(products[pid]['title'], score, summary)