#!/usr/bin/env python3
"""Precompute "similar highlights" for every book in books.json.

Each highlight is turned into a meaning vector with a small local embedding
model (BAAI/bge-small-en-v1.5, via fastembed). For each book, every highlight
is compared with every highlight in every other book; for each other book we
keep its single closest highlight, then keep the top few books. The result is
written to related.json, which the site reads at runtime, so visitors' browsers
never do this work.

Setup (once, outside the repo):
    python3 -m venv ~/.venvs/related
    ~/.venvs/related/bin/pip install fastembed numpy

Usage (from the repo root):
    ~/.venvs/related/bin/python scripts/build_related.py

Embeddings are cached in ~/.cache/bipul-related/ keyed by text, so after the
first run only new or edited highlights are embedded.
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ROOT / "books.json"
OUT = ROOT / "related.json"
CACHE_DIR = Path.home() / ".cache" / "bipul-related"
MODEL = "BAAI/bge-small-en-v1.5"

TOP_BOOKS = 3              # related books shown per book
MIN_SCORE = 0.72           # cosine similarity below this is not a real match (scores run ~0.66-0.90)
PREFER_MIN, PREFER_MAX = 70, 260  # readable quote length, in characters
LONG_PENALTY = 0.03        # ranks over-long or very short quotes lower without excluding them
QUOTE_MAX = 240            # quotes are clipped to this length in the output


def plain(text):
    """Strip HTML tags and the leading [context:] tag used on some highlights."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\[[^\]]{3,110}:\]\s*", "", text.strip())
    return re.sub(r"\s+", " ", text).strip()


def clip(text, max_len=QUOTE_MAX):
    """Shorten to the last full sentence under max_len, or to a word boundary with an ellipsis."""
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[:end + 1] if end > 80 else cut[:cut.rfind(" ")] + "…"


def embed_all(texts):
    """Return unit-length embeddings, reusing cached vectors for text seen before."""
    try:
        from fastembed import TextEmbedding
    except ImportError:
        sys.exit("fastembed is not installed. See the setup steps at the top of this file.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "embeddings.npz"
    cached = {}
    if cache_file.exists():
        data = np.load(cache_file)
        cached = dict(zip(data["keys"].tolist(), data["vectors"]))

    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    missing = list({k: t for k, t in zip(keys, texts) if k not in cached}.items())
    if missing:
        model = TextEmbedding(MODEL, cache_dir=str(CACHE_DIR / "models"))
        vectors = np.array(list(model.embed([t for _, t in missing], batch_size=16)))
        for (k, _), v in zip(missing, vectors):
            cached[k] = v
        np.savez(cache_file, keys=np.array(list(cached.keys())), vectors=np.array(list(cached.values())))

    matrix = np.array([cached[k] for k in keys], dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True), len(missing)


def main():
    started = time.time()
    books = json.loads(BOOKS.read_text(encoding="utf-8"))

    # One record per text highlight. HTML figures have no text to compare, so they are skipped.
    items = []  # (book_index, highlight_index, text)
    for bi, book in enumerate(books):
        for hi, h in enumerate(book.get("highlights", [])):
            if h.strip().startswith("<"):
                continue
            text = plain(h)
            if text:
                items.append((bi, hi, text))

    vectors, new_count = embed_all([t for _, _, t in items])
    owner = np.array([bi for bi, _, _ in items])
    lengths = np.array([len(t) for _, _, t in items])
    penalty = np.where((lengths < PREFER_MIN) | (lengths > PREFER_MAX), LONG_PENALTY, 0.0)

    similarity = vectors @ vectors.T

    related = {}
    for bi, book in enumerate(books):
        mine = np.where(owner == bi)[0]
        picks = []
        if len(mine):
            best = []  # (adjusted score, raw score, source index, target index, other book)
            for other in range(len(books)):
                if other == bi:
                    continue
                theirs = np.where(owner == other)[0]
                if len(theirs) == 0:
                    continue
                block = similarity[np.ix_(mine, theirs)] - penalty[theirs][None, :]
                r, c = np.unravel_index(block.argmax(), block.shape)
                raw = float(similarity[mine[r], theirs[c]])
                best.append((float(block[r, c]), raw, mine[r], theirs[c], other))
            best.sort(reverse=True)
            for _, raw, src, tgt, other in best:
                if raw < MIN_SCORE:
                    continue
                picks.append({
                    "title": books[other]["title"],
                    "slug": re.sub(r"[^a-z0-9]+", "-", books[other]["title"].lower()).strip("-"),
                    "author": books[other].get("author", ""),
                    "cover": books[other].get("cover", ""),
                    "color": books[other].get("color", ""),
                    "quote": clip(items[tgt][2]),
                    "score": round(raw, 3),
                    "fromHighlight": int(items[src][1]),  # index in this book's highlights
                    "toHighlight": int(items[tgt][1]),    # index in the other book's highlights
                })
                if len(picks) == TOP_BOOKS:
                    break
        related[book["title"]] = picks

    OUT.write_text(json.dumps(related, ensure_ascii=False, indent=1), encoding="utf-8")
    filled = sum(1 for v in related.values() if v)
    print(f"{len(books)} books, {len(items)} highlights ({new_count} newly embedded), "
          f"{filled} books with suggestions, {time.time() - started:.1f}s -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
