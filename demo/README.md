# Demo — Interactive Routing Visualizer

Streamlit app for exploring compression routing decisions across PISCO, OSCAR, and xRAG.

## Modes

### Single

Run a single model on a query. Shows the full pipeline step by step:
- Context chunking and compression into memory tokens
- CLF probability and router verdict (compressed vs full)
- Decoder output with correctness check against gold
- Token savings breakdown

![Single mode](../pics/single_page.png)

### Comparison

Run two models side-by-side on the same context and query. Useful for comparing how different compression architectures handle the same input — routing decisions, token counts, and answer quality.

![Comparison mode](../pics/comparison_page.png)

### Scale

Batch evaluation over 1000 samples. Shows:
- Compressed vs full routing distribution at the current threshold
- Token savings percentage and router accuracy
- Accuracy vs tokens/query tradeoff plot (compress-all, router, full-context)

![Scale mode](../pics/scale_page.png)

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app tries to load real models from local paths (see `MODEL_PATH` in `app.py`). If a model isn't available, it falls back to `MockModel` which returns random outputs — useful for testing the UI without GPU.

## Files

- `app.py` — main Streamlit application
- `mock_model.py` — mock model for UI development without GPU
- `presets.json` — preset context/query pairs for quick testing
- `scale_data.json` — pre-computed batch results for scale mode

## Model paths

Update `MODEL_PATH` in `app.py` to point to your local model directories or HuggingFace repos:

```python
MODEL_PATH = {
    "PISCO": "/models/pisco-router",
    "OSCAR": "/models/oscar-mistral-7B",
    "xRAG": "Hannibal046/xrag-7b",
}
```
