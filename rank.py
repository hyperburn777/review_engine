import math
import re

import numpy as np
import spacy
from nltk.stem import PorterStemmer
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Cross-Encoder as a reranker on top of similarities
reranker_model = CrossEncoder(RERANKER_MODEL, max_length=512)


def compute_reranker_scores(query, products, pairs, model, batch_size=32):
    if not pairs:
        return np.array([], dtype=float)

    query_2_products_text = []
    for pid, _ in pairs:
        meta_data = products.get(pid, {}) or {}
        title = meta_data.get("title") or ""
        summary = meta_data.get("summary") or ""
        overall_description = (title + " " + summary).strip()

        if not overall_description:
            overall_description = ""

        query_2_products_text.append((query, overall_description))

    scores = []
    for i in range(0, len(query_2_products_text), batch_size):
        batch = query_2_products_text[i:i + batch_size]
        batch_scores = model.predict(batch)

        if hasattr(batch_scores, "tolist"):
            batch_scores = batch_scores.tolist()

        scores.extend(batch_scores)

    return np.array(scores, dtype=float)


def min_max_normalization(values, lower=1, upper=99, epsilon=1e-9):
    arr = np.array(values, dtype=float)

    low, high = np.percentile(arr, [lower, upper])

    clipped = np.clip(arr, low, high)

    if np.isclose(high, low):
        return np.ones_like(arr) * 0.5

    return (clipped - low) / (high - low + epsilon)


MATCH_TEXT_2_INT_PATTERN = r"^(\d+(\.\d+)?)(k|kk|k\+)?$"
MATCH_WITH_2_BOUNDS_PATTERN = r'(?:between\s+)?\$?(\d[\d,\.]*\s*(?:k|kk)?)\s*(?:-+|~|to|and)\s*\$?(\d[\d,\.]*\s*(?:k|kk)?)'
MATCH_WITH_UPPER_BOUND_PATTERN = r'(?:under|below|less than|at most|≤|&lt;=)\s*\$?(\d[\d,\.]*\s*(?:k|kk)?)'
MATCH_APPROX_PATTERN = r'(?:around|about|approx(?:imately)?|near|nearly)\s*\$?(\d[\d,\.]*\s*(?:k|kk)?)'
MATCH_EXP_PATTERN_1 = r'\$?(\d[\d,\.]*\s*(?:k|kk)?)\s*(?:budget|budgets?)'
MATCH_EXP_PATTERN_2 = r'(?:budget|budgets?)\s*\$?(\d[\d,\.]*\s*(?:k|kk)?)'
THOUSAND = 1000.0


def parse_budget(query):
    # Extract anchor budget number

    lower_query = query.lower()

    def parse_money(text):
        text = text.replace(",", "").strip()
        regex_match = re.match(MATCH_TEXT_2_INT_PATTERN, text)

        if not regex_match:
            return None

        val = float(regex_match.group(1))
        if regex_match.group(3):
            val *= THOUSAND

        return val

    # Range: 100-200 / 100~200 / between 100 and 200
    regex_match = re.search(MATCH_WITH_2_BOUNDS_PATTERN, lower_query)

    if regex_match:
        bound_a = parse_money(regex_match.group(1))
        bound_b = parse_money(regex_match.group(2))

        if bound_a is not None and bound_b is not None:
            return (bound_a + bound_b) / 2.0

    # Upper bound: under/below/less than/<=
    regex_match = re.search(MATCH_WITH_UPPER_BOUND_PATTERN, lower_query)

    if regex_match:
        money = parse_money(regex_match.group(1))

        if money is not None:
            return money

    # Approximation: around/about/approx/nearly
    regex_match = re.search(MATCH_APPROX_PATTERN, lower_query)

    if regex_match:
        money = parse_money(regex_match.group(1))

        if money is not None:
            return money

    # Number: $500 budget / budget 800
    regex_match = re.search(MATCH_EXP_PATTERN_1, lower_query)

    if regex_match:
        money = parse_money(regex_match.group(1))

        if money is not None:
            return money

    regex_match = re.search(MATCH_EXP_PATTERN_2, lower_query)

    if regex_match:
        money = parse_money(regex_match.group(1))

        if money is not None:
            return money

    return None


stemmer = PorterStemmer()


def stem(word):
    lw = word.lower()

    stemmed = stemmer.stem(lw)
    if len(stemmed) < 3:
        return lw
    return stemmed


tokenization_model = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def tokenize(text):
    if not text:
        return []

    tokenized = tokenization_model(text)
    tokens = []

    for token in tokenized:
        if token.is_punct or token.is_stop:
            continue

        tokens.append(token.lemma_.lower())

    return tokens


def price_fit_score(price, anchor, price_std=None,
                    base_ratio=0.3, std=0.5, over_weight=0.6, under_weight=1.2):
    if anchor is None or anchor <= 0:
        return 0.5

    # No price / dirty value
    # Treat it as a risky product, return a low score
    if price is None or price <= 0:
        return 0.2

    sigma_base = base_ratio * anchor
    if price_std is not None and price_std > 0:
        sigma_base = max(sigma_base, std * float(price_std))

    sigma = (over_weight if price > anchor else under_weight) * sigma_base

    return math.exp(-((price - anchor) ** 2) / (2 * sigma ** 2))


def bayesian_rating(product_rate, product_rate_count, global_avg_rate=4.3, m=100.0):
    if product_rate is None:
        product_rate = global_avg_rate

    if product_rate_count is None:
        product_rate_count = 0

    return (global_avg_rate * m + product_rate * product_rate_count) / (m + product_rate_count)


MAX_STAR_RATING = 5.0


def mmr_cosine_similarity(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return 0.0

    arr_a = np.asarray(vec_a, dtype=float)
    arr_b = np.asarray(vec_b, dtype=float)

    return float(np.dot(arr_a, arr_b))


EPSILON = 1e-9


def rank(
        product_ids, similarities, products, reviews, query,
        top_k_candidates=200, final_k=10,
):
    # Desc by cos sim, pick up top-k
    pairs = sorted(zip(product_ids, similarities), key=lambda x: x[1], reverse=True)[:top_k_candidates]

    if not pairs:
        return []

    # Reranking
    reranker_scores = compute_reranker_scores(query, products, pairs, model=reranker_model)
    normalized_reranker_scores = min_max_normalization(reranker_scores)

    # Calculate price anchor
    budget = parse_budget(query)

    candidate_prices = []
    for pid, _ in pairs:
        price = products.get(pid, {}).get('price')
        if price is not None:
            candidate_prices.append(price)

    price_anchor = budget if budget else (float(np.median(candidate_prices)) if candidate_prices else None)
    price_std = (float(np.std(candidate_prices)) if candidate_prices else None)

    # Compute features of each product
    ratings = []
    popularities = []
    price_fits = []

    for pid, sim in pairs:
        meta_data = products.get(pid, {})

        star_rating = meta_data.get('rating')
        review_count = meta_data.get('review_count')
        price = meta_data.get('price')

        bayesian_rating_score = bayesian_rating(star_rating, review_count) / MAX_STAR_RATING

        popularity = math.log1p(review_count) if isinstance(review_count, (int, float)) and review_count >= 0 else 0.0

        price_fit = price_fit_score(price, price_anchor, price_std=price_std)

        ratings.append(bayesian_rating_score)
        popularities.append(popularity)
        price_fits.append(price_fit)

    pop_norm = min_max_normalization(popularities)

    # Default weights
    weights = {
        "alpha": 0.4,  # cos similarity
        "beta": 0.2,  # bayesian rating
        "gamma": 0.1,  # popularity
        "delta": 0.15,  # price fit
        "epsilon": 0.15,  # reranker score
        "mmr_lambda": 0.66  # MMR
    }

    # Merging
    final_rating_scores = []

    for i, (pid, sim) in enumerate(pairs):
        score = (
                weights["alpha"] * float(sim) +
                weights["beta"] * float(ratings[i]) +
                weights["gamma"] * float(pop_norm[i]) +
                weights["delta"] * float(price_fits[i]) +
                weights["epsilon"] * float(normalized_reranker_scores[i])
        )
        final_rating_scores.append((pid, score))

    # MMR
    id_2_vec = {}

    for pid, _ in pairs:
        review_vector = reviews.get(pid)
        arr = np.asarray(review_vector, dtype=float)
        vec_norm = np.linalg.norm(arr)

        id_2_vec[pid] = arr / (vec_norm + EPSILON)

    id_2_score = dict(final_rating_scores)
    mmr_lambda = float(weights.get("mmr_lambda"))

    selected = []
    candidates = [pid for pid, _ in sorted(final_rating_scores, key=lambda x: x[1], reverse=True)]

    while len(selected) < min(final_k, len(candidates)):
        best_pid = None
        best_mmr_score = -1e9

        for pid in candidates:
            relevance = id_2_score.get(pid, 0.0)

            if not selected:
                mmr_val = relevance
            else:
                similarities_to_selected = [
                    mmr_cosine_similarity(id_2_vec.get(pid), id_2_vec.get(s))
                    for s in selected
                ]
                max_sim = max(similarities_to_selected) if similarities_to_selected else 0.0

                mmr_val = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim

            if mmr_val > best_mmr_score:
                best_pid = pid
                best_mmr_score = mmr_val

        if best_pid is None:
            break

        selected.append(best_pid)
        candidates.remove(best_pid)

    final = [(pid, id_2_score[pid]) for pid in selected]

    with_price = []
    without_price = []

    for pid, score in final:
        price = products.get(pid, {}).get("price")
        if price is None:
            without_price.append((pid, score))
        else:
            with_price.append((pid, score))

    return with_price + without_price
