"""
MockModel — drop-in mock for COCOMRouter (modelling_pisco_router.py).

Pipeline order (matches real model):
  1. chunk text
  2. compress chunks → memory embeddings
  3. router CLF on compressed hidden states → decide compressed vs full
  4. generate with chosen path
"""

import functools
import hashlib
import logging
import queue
import random
import re
import time
from collections import Counter

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

FIRST_TOKEN_DELAY = {"PISCO": 0.08, "OSCAR": 0.045, "xRAG": 0.025}
DECODE_STEP = 0.019
CHUNK_SIZE = 128


def _tokenize(text):
    return text.split()


def _stable_clf_prob(text, query, model_name=""):
    """Deterministic pseudo-CLF for free-form inputs (no saved preset).

    A real router classifier is deterministic — same input, same probability —
    so we hash the (query, context, model) instead of drawing a fresh random
    value each rerun (which made the CLF badge flicker). Range [0.05, 0.95]."""
    key = f"{model_name}␟{query}␟{text}".encode("utf-8")
    h = int(hashlib.md5(key).hexdigest()[:8], 16)
    return round(0.05 + (h % 9001) / 10000, 4)


def _sent_split(text):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sents if s]


def _f1(pred_tokens, gold_tokens):
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    p = n_common / len(pred_tokens)
    r = n_common / len(gold_tokens)
    return 2 * p * r / (p + r)


def _best_sentences(context, query, max_sents=2):
    sents = _sent_split(context)
    if not sents:
        return context[:200]
    q_toks = [w.lower() for w in _tokenize(query)]
    scored = []
    for s in sents:
        s_toks = [w.lower() for w in _tokenize(s)]
        scored.append((_f1(s_toks, q_toks), s))
    scored.sort(key=lambda x: -x[0])
    return " ".join(s for _, s in scored[:max_sents])


# Small sentence-embedding model (~90 MB, CPU-friendly) for the free-form answer.
_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@functools.lru_cache(maxsize=1)
def _sentence_encoder():
    """(tokenizer, model), loaded once per process; None if unavailable (e.g. offline)."""
    try:
        from transformers import AutoModel, AutoTokenizer

        return (
            AutoTokenizer.from_pretrained(_EMBED_MODEL),
            AutoModel.from_pretrained(_EMBED_MODEL).eval(),
        )
    except Exception as e:  # any load failure → lexical fallback, never a crash
        log.warning("Sentence encoder %s unavailable (%s); using word overlap", _EMBED_MODEL, e)
        return None


@torch.no_grad()
def _embed(texts, tok, model, batch_size=64):
    """Mean-pooled, L2-normalized sentence embeddings, shape (len(texts), d)."""
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i : i + batch_size], padding=True, truncation=True,
                  max_length=256, return_tensors="pt")
        hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        out.append((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9))
    return F.normalize(torch.cat(out), dim=-1)


def _closest_sentence(context, query):
    """The context sentence semantically closest to the query (cosine similarity)."""
    sents = _sent_split(context)
    if not sents:
        return context[:200]
    encoder = _sentence_encoder()
    if encoder is None:
        return _best_sentences(context, query, max_sents=1)
    embs = _embed([query] + sents, *encoder)
    return sents[int((embs[1:] @ embs[0]).argmax())]


class MockModel:
    """Drop-in mock for COCOMRouter.

    Usage:
        model = MockModel("PISCO")
        result = model.run_pipeline(context, query)

        for update in model.run_pipeline_with_progress(context, query):
            ...
    """

    # Operating thresholds from new_presets.jsonl (so the default route matches).
    DEFAULT_THRESHOLD = {"xRAG": 0.5566, "PISCO": 0.1992, "OSCAR": 0.1325}

    def __init__(self, model_name="PISCO", num_mem_tokens=None, d_model=4096):
        self.model_name = model_name
        self.num_mem_tokens = (
            num_mem_tokens or {"xRAG": 1, "PISCO": 8, "OSCAR": 8}[model_name]
        )
        self.d_model = d_model
        self.doc_max_length = CHUNK_SIZE
        self.routing_threshold = self.DEFAULT_THRESHOLD[model_name]

    def to(self, *args, **kwargs):
        return self

    # No-op GPU hooks so _ensure_gpu can call these unconditionally (the mock
    # has no real weights to move).
    def park_gpu(self):
        return self

    def unpark_gpu(self):
        return self

    def compress(self, input_ids, attention_mask):
        """(n_chunks, seq_len) → (n_chunks, num_mem_tokens, d_model)
        xRAG: always 1 total token regardless of chunks."""
        n_chunks = input_ids.shape[0]
        if self.model_name == "xRAG":
            return torch.randn(1, 1, self.d_model)
        return torch.randn(n_chunks, self.num_mem_tokens, self.d_model)

    def prepare_encoder_inputs(self, chunks, max_length=128):
        n = len(chunks)
        return {
            "input_ids": torch.randint(0, 32000, (n, max_length)),
            "attention_mask": torch.ones(n, max_length, dtype=torch.long),
        }

    def run_pipeline_with_progress(
        self, text, query=None, mode="auto", threshold=0.5, preset=None,
    ):
        """Generator yielding pipeline stages.

        preset: merged dict with keys answer, full_answer, clf_prob,
                tokens_compressed, tokens_full. None when running free-form.
        """

        words = _tokenize(text)
        tokens_full = len(words)
        n_chunks = max(1, (tokens_full + CHUNK_SIZE - 1) // CHUNK_SIZE)

        t0 = time.perf_counter()
        enc = self.prepare_encoder_inputs(
            [f"chunk_{i}" for i in range(n_chunks)],
            max_length=self.doc_max_length,
        )
        compressed_embs = self.compress(enc["input_ids"], enc["attention_mask"])
        time.sleep(0.08)

        yield {
            "stage": "compression",
            "compressed_embs": compressed_embs,
            "tokens_full": tokens_full,  # context tokens (no prompt), for the bars
            "exec_time": time.perf_counter() - t0,
        }

        t0 = time.perf_counter()
        clf_prob = preset["clf_prob"] if preset else _stable_clf_prob(text, query, self.model_name)
        if mode == "auto":
            chosen = "compressed" if clf_prob <= threshold else "full"
        else:
            chosen = mode
        time.sleep(0.09)

        yield {
            "stage": "router",
            "clf_prob": clf_prob,
            "mode": chosen,
            "exec_time": time.perf_counter() - t0,
        }

        yield {"stage": "generation"}

        if preset:
            prediction = preset.get("full_answer") if chosen == "full" else preset.get("answer")
            prediction = prediction or preset.get("answer", "")
        else:
            q = query if query else text[:100]
            prediction = _closest_sentence(text, q)

        answer_words = prediction.split()
        t0 = time.perf_counter()
        time.sleep(FIRST_TOKEN_DELAY[self.model_name])
        for i, word in enumerate(answer_words):
            yield {"stage": "token", "token": (" " if i > 0 else "") + word}
            if i < len(answer_words) - 1:
                time.sleep(DECODE_STEP)

        # Context-only counts (no prompt): full = context tokens, compressed = mem tokens.
        tokens_compressed = compressed_embs.shape[0] * compressed_embs.shape[1]
        tokens_saved = (
            max(0, tokens_full - tokens_compressed) if chosen == "compressed" else 0
        )

        yield {
            "stage": "done",
            "result": {
                "prediction": prediction,
                "mode": chosen,
                "clf_prob": round(clf_prob, 4),
                "tokens_full": tokens_full,
                "tokens_compressed": tokens_compressed,
                "tokens_saved": tokens_saved,
                "exec_time": round(time.perf_counter() - t0, 3),
            },
        }

    def run_pipeline(self, text, query=None, **kwargs):
        for update in self.run_pipeline_with_progress(text, query, **kwargs):
            if update["stage"] == "done":
                return update["result"]



class MockStreamer:
    """Iterator that receives tokens via put() from another thread."""

    def __init__(self):
        self._q = queue.Queue()
        self._done = False

    def put(self, token):
        self._q.put(token)

    def end(self):
        self._q.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._q.get()
        if item is None:
            raise StopIteration
        return item
