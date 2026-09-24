# OverflowGuard Uncertainty Baselines

This folder contains uncertainty-estimation baselines for OverflowGuard.

The intended comparison is:

## Pre-generation baselines

Computed from input tokens after the **compressed context**:

- Perplexity
- Focus
- RAUQ

These use the estimator implementations from
[`IINemo/lm-polygraph`](https://github.com/IINemo/lm-polygraph).

# 1. Files

```text
uncertainty_common.py
    Shared utilities:
      - lm-polygraph estimator construction
      - attention/log-prob statistics
      - JSONL helpers

collect_ue_xrag.py
    Computes post-generation uncertainty for xRAG compressed answers.

collect_ue_pisco.py
    Computes post-generation uncertainty for PISCO compressed answers.

collect_ue_oscar.py
    Computes post-generation uncertainty for PISCO compressed answers.

eval_ue_auc.py
    Once correctness labels exist, evaluates an AUC ROC score as an
    OverflowGuard classifier.

eval_ue_acc.py
    Once correctness labels exist, evaluates an accuracy score as an
    OverflowGuard router.
```

The scripts assume that OverflowGuard has already created a `collection.pt`.

Typical directory:

```text
xrag_moe_router_ckpt_squad/
├── collection.pt
├── emb_cache_train_collect.pt
└── eval/
    ├── collection.pt
    └── emb_cache_eval_collect.pt
```

or:

```text
pisco_router_ckpt/
├── collection.pt
└── eval/
    └── collection.pt
```

The collection rows should contain at least:

```python
{
    "id": ...,
    "context": ...,
    "query": ...,
    "gold": ...,
    "comp_answer": ...,
    "full_answer": ...
}
```

`comp_correct` and `full_correct` may still be `None`.

# 2. Installation

Create/activate the same Python environment you use for OverflowGuard.

Install LM-Polygraph:

```bash
pip install git+https://github.com/IINemo/lm-polygraph.git
```

Useful dependencies:

```bash
pip install numpy scipy scikit-learn tqdm transformers accelerate sentence-transformers
```

Each post-generation output row looks approximately like:

```json
{
  "id": 17,
  "query": "...",
  "perplexity": 1.83,
  "focus": 4.27,
  "rauq": 1.19,
  "answer_num_tokens": 8
}
```

The first three values are the baselines intended for evaluation.

# 3. xRAG: post-generation uncertainty

`collect_ue_xrag.py` uses the compressed answer already stored in
`collection.pt`.

It does **not** generate another answer.

Instead it teacher-forces the cached `comp_answer` through the xRAG compressed
path and obtains:

- token likelihoods
- token distributions
- attentions

These statistics are then supplied to the LM-Polygraph Perplexity, Focus and
RAUQ estimators.

# 4. PISCO: post-generation uncertainty

For PISCO, the script imports your existing `PiscoRouter` implementation from
your OverflowGuard `train_pisco.py`.

Train split:

```bash
PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES=0 python ue/collect_ue_oscar.py \
    --collection ./oscar_qwen-7b_router_ckpt_combined/collection.pt \
    --router-script ./examples/train_oscar.py \
    --model naver/oscar-qwen2-7B \
    --output ./ue_combined_oscar_qwen/train.jsonl \
    --max-samples 1000 \
    --focus-idf-path ./ue/cache/focus_oscar_qwen_combined/token_idf.pkl

PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES=0 python ue/collect_ue_xrag.py \
    --collection ./xrag_mixtral-8x7b_router_ckpt_combined/collection.pt \
    --emb-cache ./xrag_mixtral-8x7b_router_ckpt_combined/emb_cache_train_collect.pt \
    --model Hannibal046/xrag-moe \
    --xrag-repo /app/xRAG \
    --output ./ue_combined_xrag_moe/train.jsonl \
    --max-samples 1000 \
    --device-map balanced_low_0 \
    --focus-idf-path ./ue/cache/focus_xrag_moe_combined/token_idf.pkl
```

Test split:

```bash
PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES=0 python ue/collect_ue_oscar.py \
    --collection ./oscar_mistral-24b_router_ckpt_combined/eval/collection.pt \
    --router-script ./examples/train_oscar.py \
    --model naver/oscar-mistral-small-24b \
    --output ./ue_combined_oscar_24b/test.jsonl \
    --focus-idf-path ./ue/cache/focus_oscar_24b_combined_test/token_idf.pkl
```
```bash
PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES=0 python ue/collect_ue_xrag.py \
    --collection ./xrag_mixtral-8x7b_router_ckpt_combined/eval/collection.pt \
    --emb-cache ./xrag_mixtral-8x7b_router_ckpt_combined/emb_cache_eval_collect.pt \
    --model Hannibal046/xrag-moe \
    --xrag-repo /app/xRAG \
    --output ./ue_combined_xrag_moe/test.jsonl \
    --device-map balanced_low_0 \
    --focus-idf-path ./ue/cache/focus_xrag_moe_combined_test/token_idf.pkl
```

## Compute test AUC

Perplexity:

```bash
PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES=0 python ue/eval_ue_auc.py \
  --scores ./ue_combined_xrag_moe/test.jsonl \
  --collection ./xrag_mixtral-8x7b_router_ckpt_combined/eval/collection.pt \
  --field perplexity \
  --method Perplexity \
  --dataset Combined \
  --compressor xRAG \
  --model mixtral-8x7b \
  --output ./ue/results_new/xrag_combined_mistral_moe_perplexity_auc.json
```
```bash
PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES=0 python ue/eval_ue_acc.py \
  --train-scores ./ue_combined_xrag_moe/train.jsonl \
  --train-collection ./xrag_mixtral-8x7b_router_ckpt_combined/collection.pt \
  --test-scores ./ue_combined_xrag_moe/test.jsonl \
  --test-collection ./xrag_mixtral-8x7b_router_ckpt_combined/eval/collection.pt \
  --field perplexity \
  --method Perplexity \
  --dataset Combined \
  --compressor xRAG \
  --model mixtral-8x7b \
  --output ./ue/results_new/xrag_combined_mistral_moe_perplexity_acc.json
```

Focus:

```bash
PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES=0 python ue/eval_ue_auc.py \
  --scores ./ue_combined_xrag_moe/test.jsonl \
  --collection ./xrag_mixtral-8x7b_router_ckpt_combined/eval/collection.pt \
  --field focus \
  --method Focus \
  --dataset Combined \
  --compressor xRAG \
  --model mixtral-8x7b \
  --output ./ue/results_new/xrag_combined_mistral_moe_focus_auc.json
```
```bash
PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES=0 python ue/eval_ue_acc.py \
  --train-scores ./ue_combined_xrag_moe/train.jsonl \
  --train-collection ./xrag_mixtral-8x7b_router_ckpt_combined/collection.pt \
  --test-scores ./ue_combined_xrag_moe/test.jsonl \
  --test-collection ./xrag_mixtral-8x7b_router_ckpt_combined/eval/collection.pt \
  --field focus \
  --method Focus \
  --dataset Combined \
  --compressor xRAG \
  --model mixtral-8x7b \
  --output ./ue/results_new/xrag_combined_mistral_moe_focus_acc.json
```

RAUQ:

```bash
PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES=0 python ue/eval_ue_auc.py \
  --scores ./ue_combined_xrag_moe/test.jsonl \
  --collection ./xrag_mixtral-8x7b_router_ckpt_combined/eval/collection.pt \
  --field rauq \
  --method RAUQ \
  --dataset Combined \
  --compressor xRAG \
  --model mixtral-8x7b \
  --output ./ue/results_new/xrag_combined_mistral_moe_rauq_auc.json
```
```bash
PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES=0 python ue/eval_ue_acc.py \
  --train-scores ./ue_combined_xrag_moe/train.jsonl \
  --train-collection ./xrag_mixtral-8x7b_router_ckpt_combined/collection.pt \
  --test-scores ./ue_combined_xrag_moe/test.jsonl \
  --test-collection ./xrag_mixtral-8x7b_router_ckpt_combined/eval/collection.pt \
  --field rauq \
  --method RAUQ \
  --dataset Combined \
  --compressor xRAG \
  --model mixtral-8x7b \
  --output ./ue/results_new/xrag_combined_mistral_moe_rauq_acc.json
```

## Final table data
```bash
python ue/make_comparison_table.py \
  --reference ./ue/overflowguard_combined_reference.csv \
  --ue-manifest ./ue/ue_auc_acc_manifest.example.jsonl \
  --output ./ue/result_table_pisco_oscar_combined.tex \
  --only-with-ue 
```


