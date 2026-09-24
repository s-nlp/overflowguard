# OverflowGuard Uncertainty Baselines

This folder contains uncertainty-estimation baselines for OverflowGuard.

The intended comparison is:

## Post-generation baselines

Computed from the **compressed-path answer**:

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


---

# 6. Resume behavior

Both post-generation collectors write JSONL incrementally.

If the job is interrupted, simply run the same command again.

Already-written sample IDs are skipped.

This makes long experiments safe to resume.

You can inspect progress with:

```bash
wc -l ./ue/xrag_squad_train.jsonl
```

or:

```bash
tail -n 3 ./ue/xrag_squad_train.jsonl
```

---

# 7. First test on a small number of examples

Before running thousands of examples, test the pipeline:

```bash
CUDA_VISIBLE_DEVICES=0 \
python collect_ue_xrag.py \
    --collection ./xrag_moe_router_ckpt_squad/collection.pt \
    --emb-cache ./xrag_moe_router_ckpt_squad/emb_cache_train_collect.pt \
    --model Hannibal046/xrag-moe \
    --xrag-repo /app/overflow-detection/scripts/xRAG \
    --output ./ue/test_10.jsonl \
    --device-map auto \
    --focus-idf-path ./ue/cache/focus_xrag/token_idf.pkl \
    --max-samples 10
```

Then inspect:

```bash
cat ./ue/test_10.jsonl
```

Do this before launching the complete train/test collections.

---

# 8. MetaUE-style pre-generation uncertainty

MetaUE is conceptually different from Perplexity/Focus/RAUQ.

Those methods inspect properties of the generated compressed answer.

MetaUE tries to predict a generation-based uncertainty value **from the input
alone**.

The training pipeline is:

```text
compressed-path generation
        ↓
generation-based uncertainty teacher
        ↓
logit_magnitude_teacher
        ↓
frozen text encoder
        ↓
2-layer MLP
        ↓
predicted uncertainty
```

At deployment/router inference:

```text
context + query
        ↓
MetaUE
        ↓
uncertainty score
```

No answer generation is required.

## Train MetaUE

Example:

```bash
PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES=0 python ue/train_metaue.py \
    --train-data ./hotpotqa/train.jsonl \
    --train-scores ./ue_hotpotqa/train.jsonl \
    --test-data ./hotpotqa/test.jsonl \
    --output-dir ./ue_hotpotqa/metaue_llama \
    --encoder sentence-transformers/all-MiniLM-L6-v2
```

This produces:

```text
ue/metaue_xrag_squad/
├── metaue.pt
├── train_metaue_scores.jsonl
└── test_metaue_scores.jsonl
```

The output score file contains:

```json
{"id": 0, "metaue": 0.438}
{"id": 1, "metaue": 0.713}
...
```

## MetaUE

```bash
PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES=0 python ue/eval_ue_router.py \
    --train-scores ./ue_hotpotqa/metaue_llama/train_metaue_scores.jsonl \
    --train-collection ./pisco_llama_hotpotqa/collection.pt \
    --test-scores ./ue_hotpotqa/metaue_llama/test_metaue_scores.jsonl \
    --test-collection ./pisco_llama_hotpotqa/eval/collection.pt \
    --field metaue
```

## Teacher target

By default:

```bash
--teacher-field logit_magnitude_teacher
```

is used.

You can change it if you deliberately want to distill another post-generation
uncertainty measure:

```bash
--teacher-field perplexity
```

or:

```bash
--teacher-field rauq
```

For the main MetaUE-style experiment, keeping
`logit_magnitude_teacher` is preferable.

---

# 9. MetaUE input

By default MetaUE sees:

```text
Context:
<context>

Question:
<query>
```

This is controlled by:

```bash
--text-mode context_query
```

You can run a query-only ablation with:

```bash
--text-mode query
```

The context+query setting is generally the more relevant baseline for
OverflowGuard because the router is deciding whether the available compressed
context is sufficient for answering the query.

---

# 10. Important: labels are NOT required yet

You can run all of the following before your colleague grades the samples:

```text
Perplexity
Focus
RAUQ
Logit Magnitude teacher
MetaUE training
MetaUE inference
```

Correctness labels become necessary only when asking:

```text
Does uncertainty predict overflow?
```

Once your colleague fills:

```python
comp_correct = True / False
full_correct = True / False
```

you can evaluate the saved uncertainty scores without rerunning the model.

---

# 11. Evaluate uncertainty as an OverflowGuard router

The target used by `eval_ue_router.py` is:

```text
overflow = compressed answer is incorrect
```

i.e.:

```python
overflow = not comp_correct
```

The evaluator:

1. reads uncertainty scores
2. reads labels from `collection.pt`
3. fits a scalar threshold on TRAIN
4. applies the same threshold to TEST
5. reports AUROC, AUPR and routing accuracy

## Perplexity

```bash
python eval_ue_router.py \
    --train-scores ./ue/xrag_squad_train.jsonl \
    --train-collection ./xrag_moe_router_ckpt_squad/collection.pt \
    --test-scores ./ue/xrag_squad_test.jsonl \
    --test-collection ./xrag_moe_router_ckpt_squad/eval/collection.pt \
    --field perplexity
```

## Focus

```bash
python eval_ue_router.py \
    --train-scores ./ue/xrag_squad_train.jsonl \
    --train-collection ./xrag_moe_router_ckpt_squad/collection.pt \
    --test-scores ./ue/xrag_squad_test.jsonl \
    --test-collection ./xrag_moe_router_ckpt_squad/eval/collection.pt \
    --field focus
```

## RAUQ

```bash
python eval_ue_router.py \
    --train-scores ./ue/xrag_squad_train.jsonl \
    --train-collection ./xrag_moe_router_ckpt_squad/collection.pt \
    --test-scores ./ue/xrag_squad_test.jsonl \
    --test-collection ./xrag_moe_router_ckpt_squad/eval/collection.pt \
    --field rauq
```

## MetaUE

```bash
python eval_ue_router.py \
    --train-scores ./ue/metaue_xrag_squad/train_metaue_scores.jsonl \
    --train-collection ./xrag_moe_router_ckpt_squad/collection.pt \
    --test-scores ./ue/metaue_xrag_squad/test_metaue_scores.jsonl \
    --test-collection ./xrag_moe_router_ckpt_squad/eval/collection.pt \
    --field metaue
```

---

# 12. Recommended experimental matrix

For each compression model:

```text
PISCO
xRAG
```

and each dataset:

```text
SQuAD
HotpotQA
TriviaQA
```

run:

```text
Post-generation:
    Perplexity
    Focus
    RAUQ

Pre-generation:
    MetaUE-style
```

A useful results table later is:

```text
Model | Dataset | Router signal | Pre/Post generation | AUROC | AUPR | Accuracy
```

For example:

```text
PISCO | SQuAD  | OverflowGuard | pre  | ...
PISCO | SQuAD  | MetaUE        | pre  | ...
PISCO | SQuAD  | Perplexity    | post | ...
PISCO | SQuAD  | Focus         | post | ...
PISCO | SQuAD  | RAUQ          | post | ...
```

---

# 13. Interpreting score direction

Different uncertainty estimators may use different score orientations.

For one method:

```text
higher score = more uncertain
```

while another may effectively behave in the opposite direction.

`eval_ue_router.py` determines the orientation using TRAIN only.

It does not use test labels to choose score direction.

This avoids test-set leakage.

---

# 14. Focus-specific notes

Focus is more complicated than Perplexity.

It requires:

- token probabilities
- generated text
- attention
- token IDF information
- spaCy linguistic processing

The first run may therefore require building/downloading IDF-related resources.

Use a persistent path such as:

```bash
--focus-idf-path ./ue/cache/focus_xrag/token_idf.pkl
```

so the resource can be reused between datasets/runs.

If the compute node cannot access the Internet, prepare these resources
beforehand.

---

# 15. Attention errors

Focus and RAUQ require attention tensors.

Modern Transformer models often use FlashAttention/SDPA, where full attention
matrices are not materialized.

The collectors try to use eager attention where possible.

If you see an error similar to:

```text
Model returned no attentions
```

or:

```text
output_attentions=True is not supported
```

load the model with:

```python
attn_implementation="eager"
```

For custom xRAG/PISCO model code this may require adding the option in the
model's `from_pretrained()` call.

Perplexity itself does not require attentions, but Focus and RAUQ do.

---

# 16. GPU memory

These collectors do not regenerate answers, which makes them substantially
cheaper than a full OverflowGuard collection run.

However Focus/RAUQ require full attention tensors, which can still consume
significant memory.

For xRAG/Mixtral, use model sharding:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
```

with:

```bash
--device-map balanced_low_0
```

If memory is still tight:

```bash
--device-map auto
```

may produce a different placement.

The collectors currently process uncertainty examples one at a time because
attention extraction is memory-heavy.

---

# 17. Why teacher forcing is used

Your `collection.pt` already stores the exact compressed answer generated during
the original OverflowGuard collection.

Instead of generating it again, the scripts run:

```text
compressed representation
        +
cached comp_answer
        ↓
teacher-forced forward
        ↓
token log probabilities
attention
logits
```

This has two benefits:

1. The uncertainty score corresponds to the exact answer that will later be
   judged by your colleague.
2. You avoid a second expensive generation pass.

This is particularly useful because regenerated answers could otherwise differ
after model/configuration changes.

---

# 18. MetaUE caveat for the paper

The script should be described as **MetaUE-style**, not necessarily as an exact
reproduction of every engineering detail of the original paper.

The important conceptual properties preserved here are:

- generation-derived pseudo uncertainty target
- no correctness labels required for training
- frozen pretrained text encoder
- small MLP uncertainty head
- MSE regression
- input-only uncertainty at inference time

The teacher score collection is adapted to OverflowGuard's already-cached
compressed answers.

---

# 19. Suggested run order

For one model/dataset:

```text
1. Run normal OverflowGuard collection
        ↓
   collection.pt exists
   comp_answer exists
   labels may still be None

2. Run post-generation uncertainty collector
        ↓
   perplexity
   focus
   rauq
   logit_magnitude_teacher

3. Train MetaUE
        ↓
   train_metaue_scores.jsonl
   test_metaue_scores.jsonl

4. Send collection.pt to colleague for correctness grading

5. Receive collection.pt with:
       comp_correct
       full_correct

6. Run eval_ue_router.py
        ↓
   AUROC / AUPR / routing metrics

7. Compare against OverflowGuard router
```

Steps 2 and 3 do not require correctness labels.

---

# 20. Minimal smoke-test workflow

Before running all datasets:

```bash
mkdir -p ue/cache/focus_xrag
```

Run 10 examples:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python collect_ue_xrag.py \
    --collection ./xrag_moe_router_ckpt_squad/collection.pt \
    --emb-cache ./xrag_moe_router_ckpt_squad/emb_cache_train_collect.pt \
    --model Hannibal046/xrag-moe \
    --xrag-repo /app/overflow-detection/scripts/xRAG \
    --output ./ue/smoke.jsonl \
    --device-map balanced_low_0 \
    --focus-idf-path ./ue/cache/focus_xrag/token_idf.pkl \
    --max-samples 10
```

Inspect:

```bash
cat ./ue/smoke.jsonl
```

Check that:

```text
perplexity != null
focus != null
rauq != null
logit_magnitude_teacher != null
```

Then launch the complete collection.

---

# 21. Useful inspection command

```bash
python - <<'PY'
import json

p = "./ue/xrag_squad_train.jsonl"

with open(p) as f:
    for i, line in enumerate(f):
        x = json.loads(line)
        print(
            x["id"],
            "PPL=", x.get("perplexity"),
            "Focus=", x.get("focus"),
            "RAUQ=", x.get("rauq"),
            "MetaUE teacher=", x.get("logit_magnitude_teacher"),
        )
        if i == 9:
            break
PY
```

---

# Summary

You can collect uncertainty immediately even if all correctness labels are
currently `None`.

Recommended interpretation:

```text
Perplexity / Focus / RAUQ
    = post-generation uncertainty baselines

MetaUE-style
    = pre-generation input-only uncertainty baseline
```

After labels are available:

```text
uncertainty score
        ↓
threshold learned on train
        ↓
safe compressed vs overflow/full routing
```

This makes the uncertainty baselines directly comparable to OverflowGuard as a
routing method.
