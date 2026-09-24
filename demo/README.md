# Demo — Interactive Routing Visualizer

Streamlit app for exploring compression routing decisions across PISCO, OSCAR, and xRAG.

## Setup

```bash
pip install -r ../requirements.txt
streamlit run app.py
```

Run it from this folder — the theme in `.streamlit/config.toml` is picked up from the working directory.

By default the app runs on `MockModel`, so it needs no GPU and no model weights. See [Real models](#real-models) to switch.

## Modes

Pick the mode in the sidebar, along with the model and its router threshold. The threshold slider starts at each model's trained operating point; move it toward 0 to send more queries to full context, toward 1 to compress everything.

### Single

Run a single model on a query. Shows the full pipeline step by step:
- Context chunking and compression into memory tokens
- CLF probability and router verdict (compressed vs full)
- Decoder output with correctness check against gold
- Token savings breakdown

![Single mode](../pics/single_page.png)

### Compare

Run two models side-by-side on the same context and query, with both columns streaming together. Useful for comparing how different compression architectures handle the same input — routing decisions, token counts, and answer quality.

![Compare mode](../pics/comparison_page.png)

### Batch

Batch evaluation over 1000 pre-computed samples per model. Shows:
- Compressed vs full routing distribution at the current threshold
- Token savings percentage and pipeline accuracy
- Accuracy vs tokens/query tradeoff plot (compress-all, router, full-context)

Because the run is replayed from saved results, moving the threshold and re-running is instant.

![Batch mode](../pics/scale_page.png)

## Questions and context

Three ways to fill the two input fields:

- **Preset questions** — the plaque rotates through them; "Try this" loads one. "Browse all questions" picks any of them directly.
- **Your own** — type a question and paste context.
- **Web search** — "Search web for context" pulls up to 3 sources from distinct domains via DuckDuckGo. Each becomes a pill below the context box; click a pill to toggle that source in or out of the context.

Presets come from `new_presets.jsonl`, which stores each question's gold answers plus, per model, the saved compressed and full answers, the CLF probability and the token counts. When the question **and** context both still match a preset, the app replays those saved answers, so the demo is reproducible offline. Edit either field and it becomes a custom query instead, marked with a `CUSTOM QUERY` badge, since there's no gold answer to check against.

## Mock model

`MockModel` imitates the real pipeline — chunking, compression into memory tokens, a routing decision, then token-by-token streaming — without any weights:

- **Routing probability** is a hash of (model, question, context), so it's deterministic: the same input always routes the same way, and the badge doesn't flicker between reruns.
- **Answers** for a custom question are the context sentence closest in meaning to it, found with a small sentence-embedding model (`all-MiniLM-L6-v2`, ~90 MB, downloaded on first use). Offline, it falls back to word overlap.
- **Token counts** are real counts over the text you provide, so the savings bars stay meaningful.

## Real models

```bash
OG_USE_REAL=1 streamlit run app.py
```

This loads the trained routers from the Hub via `routers.py`, which defines the `OverflowRouter` subclasses (`PiscoRouter`, `OscarRouter`, `XragRouter`). Which checkpoint each model loads is set by `MODEL_PATH` in `app.py`.

The released routers all live in one repo, [`wexumin/OverflowGuard-checkpoints`](https://huggingface.co/wexumin/OverflowGuard-checkpoints), under `{compressor}/{model}/{dataset}/`. The demo's presets and batch data were produced with the `mistral-7b` / `squad` checkpoints, so those are the matching entries:

```python
MODEL_PATH = {
    "PISCO": "pisco/mistral-7b/squad",
    "OSCAR": "oscar/mistral-7b/squad",
    "xRAG": "xrag/mistral-7b/squad",
}
# loaded as: cls.from_pretrained(CHECKPOINTS_REPO, hf_local_path=MODEL_PATH[name])
```

Swap in another compressor, base model or training dataset by changing those paths. Note that the saved preset answers and thresholds still come from the `mistral-7b` / `squad` runs.

Needs a GPU and multi-GB weights. If a model fails to load, that model falls back to `MockModel` with a warning.

Only one model sits on the GPU at a time. A run takes the GPU for its duration, swapping the previous model out to CPU first; anyone else sees "Waiting for the GPU…" until it's free. In mock mode there's no lock and no queue.

xRAG's `routers.py` subclass expects the xRAG source tree at `/workspace/xRAG`; edit that path if yours differs.

## Theme

Light and dark palettes are defined in `.streamlit/config.toml`. Each viewer picks one under ⋮ → Settings (default: their OS setting), and the app's own colors follow the choice in that browser only.

## Files

- `app.py` — main Streamlit application
- `routers.py` — `OverflowRouter` subclasses for the real models
- `mock_model.py` — mock model for UI development without GPU
- `new_presets.jsonl` — preset questions with gold and saved per-model answers
- `scale_data.json` — pre-computed batch results (1000 samples per model)
- `.streamlit/config.toml` — light/dark theme palettes
