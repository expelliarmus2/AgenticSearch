import time
from typing import List

import numpy as np
import requests


class APIEncoder:
    """Encoder backed by an OpenAI-compatible /v1/embeddings HTTP service
    (e.g. vLLM serving bge-m3). Replaces the local GPU encoder so that the
    retrieval server / index builder can run on CPU-only or NPU machines.

    bge-m3 is symmetric (no query/passage prefixes), while e5-style models
    still get their prefixes when routed through the same service.
    """

    def __init__(
        self,
        api_url: str = "http://127.0.0.1:8091/v1/embeddings",
        model_name: str = "bge-m3",
        batch_size: int = 64,
        max_retries: int = 3,
        timeout: int = 120,
        normalize: bool = True,
    ):
        self.api_url = api_url
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.normalize = normalize
        self._dim = None

    @property
    def dim(self):
        return self._dim

    def _add_prefix(self, text_list: List[str], is_query: bool) -> List[str]:
        name = self.model_name.lower()
        if "e5" in name:
            prefix = "query: " if is_query else "passage: "
            return [prefix + t for t in text_list]
        if "bge" in name and "bge-m3" not in name and is_query:
            # bge v1.5 style instruction (bge-m3 is symmetric and needs none)
            return [
                "Represent this sentence for searching relevant passages: " + t
                for t in text_list
            ]
        return list(text_list)

    def _request(self, payload_texts: List[str]) -> List[List[float]]:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.api_url,
                    json={"model": self.model_name, "input": payload_texts},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                data.sort(key=lambda x: x["index"])
                return [item["embedding"] for item in data]
            except Exception as e:  # noqa: BLE001
                last_err = e
                wait = min(2 ** attempt, 8)
                print(
                    f"[APIEncoder] request failed (attempt {attempt + 1}/{self.max_retries}): {e}; "
                    f"retrying in {wait}s",
                    flush=True,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"[APIEncoder] embedding service {self.api_url} failed after "
            f"{self.max_retries} retries: {last_err}"
        )

    def encode(self, query_list, is_query: bool = True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]
        texts = self._add_prefix(query_list, is_query)

        all_embeddings: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            all_embeddings.extend(self._request(batch))

        emb = np.asarray(all_embeddings, dtype=np.float32)
        if self._dim is None:
            self._dim = emb.shape[-1]
        if self.normalize:
            norm = np.linalg.norm(emb, axis=-1, keepdims=True)
            norm[norm == 0] = 1.0
            emb = emb / norm
        return emb
