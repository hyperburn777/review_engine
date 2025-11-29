import requests
import torch, re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoProcessor, AutoModelForImageTextToText
from io import BytesIO
from PIL import Image


def extract_summary(raw: str) -> str:
    # if the whole thing is quoted, drop outer quotes
    raw = raw.strip().strip('"').strip("'")

    # grab everything after the last "ASSISTANT:"
    m = re.search(r"ASSISTANT:\s*(.*)\s*$", raw, flags=re.S)
    if m:
        summary = m.group(1).strip()
    else:
        # fallback: no label found → return the whole thing trimmed
        summary = raw.strip()

    # clean common trailing artifacts
    summary = re.sub(r"(</s>|\[END\]|\[/ASSISTANT\])\s*$", "", summary).strip()
    return summary


def load_images(records, max_images=3):
    images = []
    for record in records[:max_images]:
        url = record["large"]
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        images.append(image)
    return images


def build_messages(name, category, rating, images, tone):
    content = []

    for _ in images:
        content.append({"type": "image"})

    content.append(
        {
            "type": "text",
            "text": (
                "You are a product copywriter. Write ONE concise customer-facing blurb (15-20 words). "
                "Use only what is clearly seen plus the metadata. Include the rating as X.X/5. "
                "Avoid brand/spec guesses or hype.\n\n"
                f"Product name: {name}\n"
                f"Category: {category}\n"
                f"Average rating: {rating:.1f}/5\n"
                f"Tone: {tone}"
            ),
        }
    )

    return [{"role": "user", "content": content}]


class Summarizer:

    def __init__(self, summary_model_id="llava-hf/llava-1.5-7b-hf", device="cpu"):
        self.summary_model_id = summary_model_id
        self.device = device
        if not torch.cuda.is_available():
            self.device = "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(summary_model_id, use_fast=True)
        self.summary_model = AutoModelForImageTextToText.from_pretrained(
            summary_model_id, dtype=self.dtype, device_map="auto"
        )

    def summarize_product(self, name, category, rating, image_records, tone="concise"):
        images = load_images(image_records, 1)

        message = build_messages(name, category, rating, images, tone)

        text = self.processor.apply_chat_template(message, add_generation_prompt=True)

        inputs = self.processor(images=images, text=text, return_tensors="pt").to(
            self.device
        )

        out = self.summary_model.generate(
            **inputs, max_new_tokens=64, temperature=0.4, do_sample=False
        )

        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    def generate_summary(self, title, main_category, average_rating, image_record):
        summary = extract_summary(
            self.summarize_product(
                title, main_category, average_rating, image_record
            )
        )

        return summary
