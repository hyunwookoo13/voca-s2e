"""Small, dependency-light visual index.

For production you can replace :class:`NumpyCosineIndex` with FAISS/HNSW.
The public interface is intentionally tiny: ``add``, ``remove``, ``search``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class SearchHit:
    """One nearest-neighbor search result."""

    key: str
    score: float
    metadata: Dict[str, str]


class NumpyCosineIndex:
    """Incremental cosine-similarity index.

    This implementation is fast enough for thousands of keyframes and has no
    compiled dependency. It stores L2-normalized vectors in a dense matrix and
    does a single matrix-vector multiplication per query.
    """

    def __init__(self, dim: int):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self._keys: List[str] = []
        self._metadata: List[Dict[str, str]] = []
        self._vectors = np.zeros((0, self.dim), dtype=np.float32)
        self._key_to_row: Dict[str, int] = {}
        self._deleted: set[str] = set()

    def __len__(self) -> int:
        return len(self._keys) - len(self._deleted)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm <= 1e-8:
            return np.zeros_like(vec, dtype=np.float32)
        return (vec / norm).astype(np.float32)

    def add(self, key: str, vector: np.ndarray, metadata: Optional[Dict[str, str]] = None) -> None:
        """Add or update one vector."""
        if not key:
            raise ValueError("key must be non-empty")
        vector = self._normalize(vector)
        if vector.shape[0] != self.dim:
            raise ValueError(f"vector dim {vector.shape[0]} != index dim {self.dim}")
        metadata = dict(metadata or {})
        if key in self._key_to_row:
            row = self._key_to_row[key]
            self._vectors[row] = vector
            self._metadata[row] = metadata
            self._deleted.discard(key)
            return
        self._key_to_row[key] = len(self._keys)
        self._keys.append(key)
        self._metadata.append(metadata)
        self._vectors = np.vstack([self._vectors, vector[None, :]])

    def remove(self, key: str) -> None:
        """Lazy-remove a vector from search results."""
        if key in self._key_to_row:
            self._deleted.add(key)

    def search(self, query: np.ndarray, top_k: int = 5, min_score: float = -1.0) -> List[SearchHit]:
        """Return top-k cosine hits above ``min_score``."""
        if len(self._keys) == 0 or top_k <= 0:
            return []
        q = self._normalize(query)
        if q.shape[0] != self.dim:
            raise ValueError(f"query dim {q.shape[0]} != index dim {self.dim}")
        if not np.any(q):
            return []
        scores = self._vectors @ q
        # Over-fetch to compensate for lazy-deleted entries.
        k = min(len(scores), max(top_k * 3, top_k))
        if k == len(scores):
            rows = np.argsort(-scores)
        else:
            rows = np.argpartition(-scores, k - 1)[:k]
            rows = rows[np.argsort(-scores[rows])]
        hits: List[SearchHit] = []
        for row in rows:
            key = self._keys[int(row)]
            if key in self._deleted:
                continue
            score = float(scores[int(row)])
            if score < min_score:
                continue
            hits.append(SearchHit(key=key, score=score, metadata=dict(self._metadata[int(row)])))
            if len(hits) >= top_k:
                break
        return hits

    def update_metadata(self, key: str, metadata: Dict[str, str]) -> None:
        """Update metadata for an existing non-deleted key.

        This is used when a provisional node is merged into a confirmed
        revisit node: the visual vector stays the same, but future retrieval
        should point at the kept node id.
        """
        if key not in self._key_to_row or key in self._deleted:
            return
        row = self._key_to_row[key]
        merged = dict(self._metadata[row])
        merged.update(dict(metadata))
        self._metadata[row] = merged

    def metadata(self, key: str) -> Optional[Dict[str, str]]:
        """Return a copy of metadata for one key, if available."""
        if key not in self._key_to_row or key in self._deleted:
            return None
        return dict(self._metadata[self._key_to_row[key]])

    def compact(self) -> None:
        """Physically remove lazy-deleted rows."""
        if not self._deleted:
            return
        keep_rows = [i for i, k in enumerate(self._keys) if k not in self._deleted]
        self._vectors = self._vectors[keep_rows]
        self._keys = [self._keys[i] for i in keep_rows]
        self._metadata = [self._metadata[i] for i in keep_rows]
        self._key_to_row = {k: i for i, k in enumerate(self._keys)}
        self._deleted.clear()

    def keys(self) -> List[str]:
        return [k for k in self._keys if k not in self._deleted]
