"""
CompressRouterModule — base class for building routed context-compression models.

Subclass this, implement the four required methods, and use the training
pipeline (``compress_router.train``) to learn a routing classifier that
decides at inference time whether to serve the compressed or full answer.

Minimal example::

    class MyRouter(CompressRouterModule):
        def compress(self, documents, query=None):
            return self.model.compress_documents(documents)

        def generate_compressed(self, compressed_embs, query, **kw):
            return self.model.generate(compressed_embs, query)

        def generate_full(self, context, query, **kw):
            return self.model.generate_full(context, query)

        def extract_clf_features(self):
            return self._captured_hs["h"][:, -1, :]

Full workflow::

    # 1. Train
    router = MyRouter.from_pretrained("wexumin/pisco")
    train_router(router, config)

    # 2. Push (saves CLF + config referencing the base model)
    router.save_pretrained("./my-router")
    router.push_to_hub("user/my-pisco-router")

    # 3. Load (auto-loads base model + CLF in one call)
    router = MyRouter.from_pretrained("user/my-pisco-router")
    result = router.run_pipeline("some text", "some query")
"""

from __future__ import annotations

import json
import os
from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, HfApi
from transformers import AutoModel, AutoTokenizer

from .classifier import RouterClassifier

_ROUTER_CONFIG = "router_config.json"
_CLF_CHECKPOINT = "routing_clf.pt"


class CompressRouterModule:
    """Abstract base for any compress-then-route model.

    Provides ``self.model`` and ``self.tokenizer`` loaded automatically
    via ``from_pretrained()``. Subclasses implement 4 methods.

    Subclasses MUST implement:
        - ``compress``
        - ``generate_compressed``
        - ``generate_full``
        - ``extract_clf_features``

    Subclasses MAY override:
        - ``evaluate`` (default: EM-or-F1 > 0.5)
        - ``park_gpu`` / ``unpark_gpu`` (custom GPU memory management)
        - ``_load_model`` (custom model loading)
        - ``_chunk_text`` (default uses tokenizer)
    """

    IDLE = "idle"
    TRAIN_COLLECT = "train_collect"
    TRAIN_EVALUATE = "train_evaluate"
    TRAIN_THRESHOLD_SEARCH = "train_threshold_search"
    TRAIN_CLF = "train_clf"
    EVAL_COLLECT = "eval_collect"
    EVAL_EVALUATE = "eval_evaluate"
    EVAL_CLF = "eval_clf"
    SAVE = "save"

    STAGES = [
        IDLE, TRAIN_COLLECT, TRAIN_EVALUATE, TRAIN_THRESHOLD_SEARCH,
        TRAIN_CLF, EVAL_COLLECT, EVAL_EVALUATE, EVAL_CLF, SAVE,
    ]

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.clf: RouterClassifier | None = None
        self.routing_threshold: float = 0.5
        self.model_name: str = ""
        self._base_model_path: str | None = None
        self._captured_hs: dict = {}
        self._hook_handle = None
        self._capture_active: bool = False
        self.stage: str = "idle"
        self._current_sample_idx: int | None = None
        self._stage_cache: dict[str, dict] = {}

    @contextmanager
    def enter_stage(self, name: str):
        prev = self.stage
        self.stage = name
        self.on_stage_enter(name)
        try:
            yield
        finally:
            self.on_stage_exit(name)
            self.stage = prev

    def on_stage_enter(self, stage: str):
        pass

    def on_stage_exit(self, stage: str):
        pass

    def print_stages(self):
        for s in self.STAGES:
            marker = " ← current" if s == self.stage else ""
            cached = f" ({len(self._stage_cache[s])} cached)" if s in self._stage_cache else ""
            print(f"  {s}{marker}{cached}")

    def get_cache(self, key=None):
        cache = self._stage_cache.get(self.stage)
        if cache is None:
            return None
        if key is not None:
            return cache.get(key)
        return cache

    def set_cache(self, key, value):
        if self.stage not in self._stage_cache:
            self._stage_cache[self.stage] = {}
        self._stage_cache[self.stage][key] = value

    # ── Loading & saving ───────────────────────────────────────────

    def _load_model(self, path: str, **kwargs):
        """Load the base model + tokenizer. Override for custom loading."""
        kwargs.setdefault("trust_remote_code", True)
        self.model = AutoModel.from_pretrained(path, **kwargs).eval()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
        except Exception:
            self.tokenizer = None

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Load a router from a local dir or HF repo.

        If ``path`` contains a ``router_config.json``, loads it as a
        full router (base model from the saved path + CLF). Otherwise
        treats ``path`` as a base model path (no CLF loaded yet — for
        training).
        """
        instance = cls()

        # check if this is a router repo (has router_config.json)
        config = instance._try_load_config(path)

        if config is not None:
            # full router repo: load base model from stored path, then CLF
            base_path = config["base_model"]
            instance._base_model_path = base_path
            instance.model_name = base_path.split("/")[-1]
            instance._load_model(base_path, **kwargs)
            instance._load_clf(path, config)
        else:
            # bare model path: load model only (for training)
            instance._base_model_path = path
            instance.model_name = path.split("/")[-1]
            instance._load_model(path, **kwargs)

        return instance

    def _try_load_config(self, path: str) -> dict | None:
        """Try to load router_config.json from path. Returns None if absent."""
        try:
            cfg_path = hf_hub_download(repo_id=path, filename=_ROUTER_CONFIG)
            return json.loads(Path(cfg_path).read_text())
        except Exception:
            local = os.path.join(path, _ROUTER_CONFIG)
            if os.path.exists(local):
                return json.loads(Path(local).read_text())
            return None

    def _load_clf(self, path: str, config: dict):
        """Load CLF weights from a router repo."""
        try:
            clf_path = hf_hub_download(repo_id=path, filename=_CLF_CHECKPOINT)
        except Exception:
            clf_path = os.path.join(path, _CLF_CHECKPOINT)

        ckpt = torch.load(clf_path, map_location="cpu", weights_only=False)
        self.clf = RouterClassifier(
            d_input=ckpt["d_input"],
            hidden=ckpt.get("hidden", 512),
        )
        self.clf.load_state_dict(ckpt["state_dict"])
        self.clf.eval()
        self.routing_threshold = config.get("threshold", ckpt.get("threshold", 0.5))

        if self.model is not None:
            device = next(self.model.parameters()).device
            self.clf.to(device)

    def save_pretrained(self, output_dir: str):
        """Save router config + CLF to a directory.

        The base model is NOT copied — only a reference to its path is
        stored in ``router_config.json``.
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.clf is None:
            raise ValueError("No CLF to save — train one first.")

        # save config with base model reference
        config = {
            "base_model": self._base_model_path,
            "threshold": self.routing_threshold,
        }
        Path(os.path.join(output_dir, _ROUTER_CONFIG)).write_text(
            json.dumps(config, indent=2)
        )

        # save CLF
        torch.save(
            {
                "d_input": self.clf.d_input,
                "hidden": self.clf.hidden,
                "state_dict": self.clf.state_dict(),
                "threshold": self.routing_threshold,
            },
            os.path.join(output_dir, _CLF_CHECKPOINT),
        )

    def push_to_hub(self, repo_id: str, output_dir: str | None = None):
        """Push router to HF Hub.

        Saves to a temp dir (or ``output_dir``), then uploads
        ``router_config.json`` and ``routing_clf.pt``.
        """
        import tempfile

        save_dir = output_dir or tempfile.mkdtemp()
        self.save_pretrained(save_dir)

        api = HfApi()
        api.create_repo(repo_id, exist_ok=True)
        api.upload_file(
            path_or_fileobj=os.path.join(save_dir, _ROUTER_CONFIG),
            path_in_repo=_ROUTER_CONFIG,
            repo_id=repo_id,
        )
        api.upload_file(
            path_or_fileobj=os.path.join(save_dir, _CLF_CHECKPOINT),
            path_in_repo=_CLF_CHECKPOINT,
            repo_id=repo_id,
        )

    # ── Required methods ────────────────────────────────────────────

    @abstractmethod
    def compress(
        self, documents: list[str], query: str | None = None
    ) -> torch.Tensor:
        """Compress document chunks into memory embeddings.

        Args:
            documents: list of text chunks.
            query: optional query for query-aware compression.

        Returns:
            Tensor of shape ``(n_chunks, n_mem_tokens, d_model)`` or similar.
        """
        ...

    @abstractmethod
    def generate_compressed(
        self,
        compressed_embs: torch.Tensor,
        query: str,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> str:
        """Generate an answer using compressed context.

        Args:
            compressed_embs: output of ``compress()``.
            query: the user query.
            max_new_tokens: generation length cap.
            **kwargs: forwarded from the pipeline (e.g. streamer for live UIs).

        Returns:
            The decoded prediction string.
        """
        ...

    @abstractmethod
    def generate_full(
        self,
        context: str,
        query: str,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> str:
        """Generate an answer using the full uncompressed context.

        Args:
            context: the full document text (not chunked).
            query: the user query.
            max_new_tokens: generation length cap.
            **kwargs: forwarded from the pipeline (e.g. streamer for live UIs).

        Returns:
            The decoded prediction string.
        """
        ...

    @abstractmethod
    def extract_clf_features(
        self,
        compressed_embs: torch.Tensor | None = None,
        query: str | None = None,
    ) -> torch.Tensor:
        """Return a 1-D feature vector for the routing classifier.

        Two usage modes:
            - **Inference** (args provided): runs a decoder forward with
              compressed embeddings + query, captures mid-layer hidden state.
            - **Training** (no args): reads state already captured during
              ``generate_compressed`` (hook fired on prefill step).

        Returns:
            1-D Tensor of shape ``(d_features,)``.
        """
        ...

    # ── GPU management (override for custom swap logic) ─────────────

    def park_gpu(self):
        """Move model to GPU. Override for partial placement (e.g. xRAG)."""
        if self.model is not None:
            self.model.to("cuda")
        if self.clf is not None:
                self.clf.to("cuda")

    def unpark_gpu(self):
        """Offload model from GPU to CPU."""
        if self.model is not None:
            self.model.to("cpu")
        if self.clf is not None:
            self.clf.to("cpu")

    def count_tokens_full(self, context: str, query: str) -> int:
        """Token count for the full (uncompressed) decoder input. Override if needed."""
        return len(self.tokenizer.encode(context))

    def count_tokens_compressed(self, compressed_embs, query: str) -> int:
        """Token count for the compressed decoder input. Override if needed."""
        return compressed_embs.shape[0] if compressed_embs.dim() == 1 else compressed_embs.shape[0] * compressed_embs.shape[1]

    def eval(self):
        if self.model is not None:
            self.model.eval()
        return self

    # ── Inference pipeline ──────────────────────────────────────────

    @torch.no_grad()
    def run_pipeline_with_progress(
        self,
        text: str,
        query: str | None = None,
        mode: str = "auto",
        threshold: float | None = None,
        max_new_tokens: int = 256,
        **kwargs,
    ):
        """Full inference pipeline as a generator.

        Yields dicts with ``stage`` key:
            - ``compression``: embeddings ready
            - ``router``: CLF decision made
            - ``generation``: decoder started
            - ``done``: final result dict
        """
        from time import perf_counter

        q = query or text[:100]
        th = threshold if threshold is not None else self.routing_threshold

        t0 = perf_counter()
        chunks = self._chunk_text(text)
        compressed_embs = self.compress(chunks, query=q)
        yield {
            "stage": "compression",
            "compressed_embs": compressed_embs,
            "exec_time": perf_counter() - t0,
        }

        t0 = perf_counter()
        clf_prob = self.clf.predict(self.extract_clf_features(compressed_embs, q)).item()

        if mode == "auto":
            use_comp = clf_prob <= th
        else:
            use_comp = mode == "compressed"

        route_mode = "compressed" if use_comp else "full"
        yield {
            "stage": "router",
            "clf_prob": clf_prob,
            "mode": route_mode,
            "exec_time": perf_counter() - t0,
        }

        yield {"stage": "generation"}

        t0 = perf_counter()
        if use_comp:
            prediction = self.generate_compressed(
                compressed_embs, q, max_new_tokens=max_new_tokens, **kwargs,
            )
        else:
            prediction = self.generate_full(
                text, q, max_new_tokens=max_new_tokens, **kwargs,
            )
        exec_time = perf_counter() - t0

        tokens_full = self.count_tokens_full(text, q)
        tokens_compressed = self.count_tokens_compressed(compressed_embs, q)
        tokens_saved = max(0, tokens_full - tokens_compressed) if use_comp else 0

        yield {
            "stage": "done",
            "result": {
                "prediction": prediction,
                "mode": route_mode,
                "clf_prob": round(clf_prob, 4),
                "tokens_full": tokens_full,
                "tokens_compressed": tokens_compressed,
                "tokens_saved": tokens_saved,
                "exec_time": round(exec_time, 3),
            },
        }

    def run_pipeline(self, text: str, query: str | None = None, **kwargs) -> dict:
        """Blocking version of ``run_pipeline_with_progress``."""
        for update in self.run_pipeline_with_progress(text, query, **kwargs):
            if update["stage"] == "done":
                return update["result"]

    # ── Internals ───────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_tokens: int = 128) -> list[str]:
        """Split text into token-accurate chunks using the tokenizer."""
        if self.tokenizer is None:
            words = text.split()
            return [
                " ".join(words[i : i + chunk_tokens])
                for i in range(0, len(words), chunk_tokens)
            ] or [text]

        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        return [
            self.tokenizer.decode(ids[i : i + chunk_tokens], skip_special_tokens=True)
            for i in range(0, len(ids), chunk_tokens)
        ] or [text]
