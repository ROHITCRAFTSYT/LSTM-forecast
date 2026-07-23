"""A lightweight TF-IDF document index for grounding the chat assistant (RAG).

Deliberately embedding-free (scikit-learn TF-IDF) so retrieval needs no network and no API
key — only the *generation* step uses Claude. Indexes project docs plus run artifacts
(e.g. a forecast summary) so the assistant can answer questions grounded in both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DocChunk:
    text: str
    source: str
    score: float = 0.0


@dataclass
class DocIndex:
    """Chunk + TF-IDF index over text snippets."""

    chunks: list[DocChunk] = field(default_factory=list)
    _vectorizer: Any = None
    _matrix: Any = None

    def add_text(self, text: str, source: str, *, min_chars: int = 40) -> None:
        """Split ``text`` into paragraph-ish chunks and add them."""
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if len(para) >= min_chars:
                self.chunks.append(DocChunk(text=para, source=source))

    def add_paths(self, paths: list[str | Path], *, patterns: tuple[str, ...] = ("*.md",)) -> None:
        """Index text files found under the given files/directories."""
        for p in paths:
            path = Path(p)
            files = [path] if path.is_file() else [f for pat in patterns for f in path.rglob(pat)]
            for f in files:
                try:
                    self.add_text(f.read_text(encoding="utf-8"), source=str(f))
                except (OSError, UnicodeDecodeError):
                    continue

    def build(self) -> DocIndex:
        """Fit the TF-IDF vectoriser over the current chunks."""
        if not self.chunks:
            return self
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform([c.text for c in self.chunks])
        return self

    def search(self, query: str, k: int = 4) -> list[DocChunk]:
        """Return the top-``k`` chunks most relevant to ``query``."""
        if not self.chunks or self._vectorizer is None:
            return []
        from sklearn.metrics.pairwise import cosine_similarity

        q = self._vectorizer.transform([query])
        sims = cosine_similarity(q, self._matrix).ravel()
        order = sims.argsort()[::-1][:k]
        return [
            DocChunk(text=self.chunks[i].text, source=self.chunks[i].source, score=float(sims[i]))
            for i in order
            if sims[i] > 0
        ]
