"""Embedding backends used by graph dataset builders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def fit_tfidf_svd_embeddings(
    texts: list[str],
    embedding_dim: int,
    max_features: int,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        strip_accents="unicode",
        lowercase=True,
    )
    tfidf = vectorizer.fit_transform(texts)
    max_components = max(1, min(embedding_dim, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
    if max_components < 2:
        dense = tfidf.toarray().astype(np.float32)
    else:
        svd = TruncatedSVD(n_components=max_components, random_state=random_seed)
        dense = svd.fit_transform(tfidf).astype(np.float32)
    if dense.shape[1] > embedding_dim:
        dense = dense[:, :embedding_dim]
    if dense.shape[1] < embedding_dim:
        dense = np.pad(dense, ((0, 0), (0, embedding_dim - dense.shape[1])))
    dense = normalize(dense, norm="l2", axis=1).astype(np.float32)
    return dense, {
        "embedding_backend": "tfidf_svd",
        "embedding_dim": embedding_dim,
        "max_features": max_features,
        "tfidf_features": int(tfidf.shape[1]),
        "svd_components": int(max_components),
    }


def _cache_key(model_name: str, text: str) -> str:
    payload = f"{model_name}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def encode_qwen_embeddings(
    texts: list[str],
    model_name: str,
    cache_dir: Path,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as error:  # pragma: no cover - depends on local env.
        raise RuntimeError(
            "Qwen embeddings require sentence-transformers. "
            "Install it with `uv add sentence-transformers` or use "
            "`--embedding-backend tfidf_svd` for smoke tests."
        ) from error

    cache_dir.mkdir(parents=True, exist_ok=True)
    vectors: list[np.ndarray | None] = []
    missing_indices: list[int] = []
    missing_texts: list[str] = []
    keys = [_cache_key(model_name, text) for text in texts]
    for index, key in enumerate(keys):
        path = cache_dir / f"{key}.npy"
        if path.exists():
            vectors.append(np.load(path).astype(np.float32))
        else:
            vectors.append(None)
            missing_indices.append(index)
            missing_texts.append(texts[index])

    if missing_texts:
        model = SentenceTransformer(model_name)
        encoded = model.encode(
            missing_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        for row, index in zip(encoded, missing_indices, strict=True):
            np.save(cache_dir / f"{keys[index]}.npy", row)
            vectors[index] = row

    dense = np.vstack([vector for vector in vectors if vector is not None]).astype(np.float32)
    metadata_path = cache_dir / "metadata.json"
    metadata = {
        "embedding_backend": "qwen",
        "qwen_model": model_name,
        "embedding_dim": int(dense.shape[1]) if dense.size else 0,
        "cache_dir": str(cache_dir),
        "num_texts": len(texts),
        "num_cache_misses": len(missing_texts),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dense, metadata


def encode_texts(
    texts: list[str],
    backend: str,
    embedding_dim: int,
    max_features: int,
    random_seed: int,
    qwen_model: str,
    cache_dir: Path,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if backend == "tfidf_svd":
        return fit_tfidf_svd_embeddings(texts, embedding_dim, max_features, random_seed)
    if backend == "qwen":
        return encode_qwen_embeddings(texts, qwen_model, cache_dir, batch_size)
    raise ValueError(f"Unsupported embedding backend: {backend}")
