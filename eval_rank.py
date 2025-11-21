import numpy as np


def safe_cosine(a, b, eps=1e-9):
    """
    安全余弦相似度，避免除零。
    a, b: 1D 向量或可转成 numpy array
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def list_diversity(product_list, review_vectors, products):
    """
    计算推荐列表的“平均两两相似度”。
    - 输入: product_list = ["ASIN1","ASIN2",...]
             review_vectors = {asin: embedding_vector, ...}
             products = {asin: { ...meta... }, ...}
    - 输出: 平均相似度(越低越好，代表越不重复)

    注意：
    - 我们用 review_vectors[pid] 作为商品的表示
    - 如果某个商品没 embedding，就用全零向量兜底，防止崩。
    - 你可以把 zeros(10) 改成 zeros(384) / zeros(768) 取决于你的真实向量维度
    """
    vecs = []
    for pid in product_list:
        v = review_vectors.get(pid)
        if v is None:
            # 如果有缺embedding的商品，给一个全零，避免报错
            v = np.zeros(10)
        vecs.append(np.asarray(v, dtype=float))

    if len(vecs) < 2:
        return 0.0

    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sims.append(safe_cosine(vecs[i], vecs[j]))

    if not sims:
        return 0.0
    return float(np.mean(sims))


def avg_bayesian_quality(product_list, products):
    """
    粗略衡量推荐列表的“质量”：
    用 rating * log(1 + review_count) 作为可信度 proxy

    越高说明推出来的是高评分+大量评论的成熟好货。
    """
    vals = []
    for pid in product_list:
        meta = products.get(pid, {})
        rating = meta.get("rating", 0.0) or 0.0
        rc = meta.get("review_count", 0) or 0
        vals.append(rating * np.log1p(rc))

    if not vals:
        return 0.0
    return float(np.mean(vals))


def avg_price(product_list, products):
    """
    计算推荐列表里的平均价格（用于看是否贴近“under $200”之类的预算）。
    """
    prices = []
    for pid in product_list:
        p = products.get(pid, {}).get("price")
        if p is not None:
            prices.append(p)
    return float(np.mean(prices)) if prices else None


def build_old_rank(product_ids, similarities, top_k=10):
    """
    基准排序器：只按 cosine 相似度从高到低排序，取前 top_k。
    返回值: [(pid, sim_score), ...]
    """
    pairs = sorted(
        zip(product_ids, similarities),
        key=lambda x: x[1],
        reverse=True
    )
    return pairs[:top_k]


def compare_and_report(query, products, reviews,
                       old_pairs, new_pairs,
                       top_k=10):
    """
    对 old 排序(纯相似度) 和 new 排序(rank() 后) 做并排对比。

    输入：
    - query: 当前查询字符串
    - products: {pid: {rating, review_count, price, title, ...}}
    - reviews: {pid: embedding_vector}  用于多样性计算
    - old_pairs: [(pid, old_score), ...]  baseline 输出
    - new_pairs: [(pid, new_score), ...]  rank() 输出
    - top_k: 只看前多少个做指标和展示

    输出：
    一个 dict，不直接 print，方便调用方自由处理 / 打印 / 保存。
    结构：
    {
        "query": ...,
        "old_order": [...],
        "new_order": [...],
        "old_rows": [
            { "pid":..., "score":..., "rating":..., "review_count":..., "price":..., "title":... },
            ...
        ],
        "new_rows": [
            { "pid":..., "score":..., "rating":..., "review_count":..., "price":..., "title":... },
            ...
        ],
        "metrics": {
            "quality_old": ...,
            "quality_new": ...,
            "diversity_old": ...,
            "diversity_new": ...,
            "avg_price_old": ...,
            "avg_price_new": ...
        }
    }
    """
    # 取前 top_k 的 pid 列表
    old_list = [pid for pid, _ in old_pairs[:top_k]]
    new_list = [pid for pid, _ in new_pairs[:top_k]]

    # 算指标
    div_old = list_diversity(old_list, reviews, products)
    div_new = list_diversity(new_list, reviews, products)

    qual_old = avg_bayesian_quality(old_list, products)
    qual_new = avg_bayesian_quality(new_list, products)

    price_old = avg_price(old_list, products)
    price_new = avg_price(new_list, products)

    # 组装报告
    result = {
        "query": query,
        "old_order": old_list,
        "new_order": new_list,
        "old_rows": [],
        "new_rows": [],
        "metrics": {
            "quality_old": qual_old,
            "quality_new": qual_new,
            "diversity_old": div_old,
            "diversity_new": div_new,
            "avg_price_old": price_old,
            "avg_price_new": price_new,
        }
    }

    # baseline 的明细
    for pid, score in old_pairs[:top_k]:
        meta = products.get(pid, {})
        result["old_rows"].append({
            "pid": pid,
            "score": float(score),
            "rating": meta.get("rating"),
            "review_count": meta.get("review_count"),
            "price": meta.get("price"),
            "title": meta.get("title", ""),
        })

    # 新 rank() 的明细
    for pid, score in new_pairs[:top_k]:
        meta = products.get(pid, {})
        result["new_rows"].append({
            "pid": pid,
            "score": float(score),
            "rating": meta.get("rating"),
            "review_count": meta.get("review_count"),
            "price": meta.get("price"),
            "title": meta.get("title", ""),
        })

    return result
