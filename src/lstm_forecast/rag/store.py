"""Vector index for analog windows: FAISS when available, NumPy brute-force otherwise.

The NumPy fallback keeps RAG fully functional (and tested) without the optional
``faiss-cpu`` dependency — important for CI and the offline smoke run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class AnalogStore:
    """A small k-NN store of window embeddings with attached "next value" payloads."""

    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None
        self._payload: np.ndarray | None = None  # next-value after each window
        self._faiss_index = None
        self._use_faiss = False

    @property
    def size(self) -> int:
        return 0 if self._embeddings is None else self._embeddings.shape[0]

    @property
    def dim(self) -> int:
        return 0 if self._embeddings is None else self._embeddings.shape[1]

    def build(self, embeddings: np.ndarray, payload: np.ndarray) -> AnalogStore:
        """Index ``embeddings`` (n, d) with aligned ``payload`` (n,)."""
        self._embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._payload = np.asarray(payload, dtype=np.float32).ravel()
        try:
            import faiss

            index = faiss.IndexFlatL2(self._embeddings.shape[1])
            index.add(self._embeddings)
            self._faiss_index = index
            self._use_faiss = True
        except ImportError:
            self._use_faiss = False
        return self

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(indices, distances, payloads)`` for the ``k`` nearest windows.

        ``query`` may be 1-D (single) or 2-D (batch). Output is always 2-D ``(n_query, k)``.
        """
        if self._embeddings is None or self._payload is None:
            raise RuntimeError("AnalogStore is empty; call build() first.")
        q = np.atleast_2d(np.asarray(query, dtype=np.float32))
        k = min(k, self.size)
        if self._use_faiss and self._faiss_index is not None:
            dists, idx = self._faiss_index.search(np.ascontiguousarray(q), k)
        else:
            # Brute-force squared L2.
            d2 = ((q[:, None, :] - self._embeddings[None, :, :]) ** 2).sum(axis=2)
            idx = np.argsort(d2, axis=1)[:, :k]
            dists = np.take_along_axis(d2, idx, axis=1)
        payloads = self._payload[idx]
        return idx, dists, payloads

    # ----------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        if self._embeddings is None or self._payload is None:
            raise RuntimeError("AnalogStore is empty; call build() before save().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Persist as a NumPy .npz (arrays only). Avoids pickle, whose loader can
        # execute arbitrary code when reading a tampered file (CWE-502).
        with path.open("wb") as fh:
            np.savez(fh, embeddings=self._embeddings, payload=self._payload)

    @classmethod
    def load(cls, path: str | Path) -> AnalogStore:
        # allow_pickle stays False (the default) so only plain arrays are read;
        # a malicious file cannot trigger code execution on load.
        with np.load(Path(path), allow_pickle=False) as data:
            embeddings = data["embeddings"]
            payload = data["payload"]
        return cls().build(embeddings, payload)
