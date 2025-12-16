# amazon_reviews
Using the amazon reviews 2023 database to design a recommendation system and q &amp; a system for users
```python
product_ids = list(reviews.keys())

emb_matrix = np.stack([reviews[pid] for pid in product_ids])

similarities = cosine_similarity([query_embedding], emb_matrix)[0]

new_pairs = rank(
    product_ids,
    similarities,
    products,
    reviews,
    query,
    top_k_candidates=200,
    final_k=10
)
```
