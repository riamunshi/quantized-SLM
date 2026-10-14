# CS F425 Deep Learning — Term Project Code

## Project Title
Recovering Reasoning Fidelity in Quantized Small Language Models
Through Inference-Time Techniques

## Hardware Used
- GPU: NVIDIA T4 (Google Colab free tier)
- VRAM: 15 GB
- All experiments run on Google Colab free tier sessions of 4-5 hours

## Environment Setup

### Python Version
Python 3.12 (Google Colab default)

### Install Dependencies
```bash
pip install transformers>=4.40.0 datasets>=2.19.0 accelerate>=0.29.0 \
            bitsandbytes>=0.43.0 peft>=0.10.0 trl>=0.8.6 pandas matplotlib numpy
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

### HuggingFace Authentication
LLaMA 1B is a gated model and requires a HuggingFace token with
Meta LLaMA access approved at:
https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

```python
from huggingface_hub import login
login(token="your_hf_token_here")
```

Qwen2 1.5B is public and requires no authentication.

---

## Folder Structure

```
03_code/
  README.md                          — this file
  requirements.txt                   — all dependencies with versions
  src/
    answer_extraction.py             — answer extraction utilities
    first_error_step_detection.py    — cascade rate computation
  scripts/
    qwen2_gsm8k_inference.ipynb      — Qwen2 GSM8K greedy + SC
    llama_gsm8k_inference.ipynb      — LLaMA GSM8K greedy + SC
    qwen2_math_inference.ipynb       — Qwen2 MATH greedy only
    llama_math_inference.ipynb       — LLaMA MATH greedy + SC
    speculative_decoding.ipynb       — speculative decoding both configs
    grpo_training.ipynb              — GRPO fine-tuning Qwen2 W8A16
    grpo_evaluation.ipynb            — GRPO evaluation on test set
    cascade_analysis.ipynb           — Priority 1 cascade analysis
    consistency_analysis.ipynb       — Priority 2 problem consistency
  configs/
    inference_config.json            — all inference hyperparameters
    grpo_config.json                 — GRPO training hyperparameters
```

---

## Running the Experiments

### 1. Qwen2 GSM8K Inference
Open `scripts/qwen2_gsm8k_inference.ipynb` in Google Colab.
Set `QUANT_TYPE` in Cell 5 to `'bf16'`, `'w8a16'`, or `'w4a16'`.
Run all cells. Results save to `/content/` as CSV.
Run once per quantization level (3 sessions total).

### 2. LLaMA GSM8K Inference
Open `scripts/llama_gsm8k_inference.ipynb` in Google Colab.
Add HuggingFace token in Cell 2.
Set `QUANT_TYPE` in Cell 5 to `'bf16'`, `'w8a16'`, or `'w4a16'`.
Run all cells. Results save to `/content/` as CSV.
Run once per quantization level (3 sessions total).

### 3. Qwen2 MATH Inference
Open `scripts/qwen2_math_inference.ipynb` in Google Colab.
Set `QUANT_TYPE` in Cell 5 to `'bf16'`, `'w8a16'`, or `'w4a16'`.
Run all cells. Results save to `/content/` as CSV.
Run once per quantization level (3 sessions total).

### 4. LLaMA MATH Inference
Open `scripts/llama_math_inference.ipynb` in Google Colab.
Add HuggingFace token in Cell 2.
Set `QUANT_TYPE` in Cell 5 to `'w8a16'` or `'w4a16'`.
For BF16, use the 4-account split notebooks in the same folder.
Run all cells. Results save to Drive automatically.

### 5. Speculative Decoding
Open `scripts/speculative_decoding.ipynb` in Google Colab.
Run all cells — both verifier configurations run automatically
back to back. Results save to Drive after every problem.
Estimated time: 2.5 hours for both configs on T4.

### 6. GRPO Training
Open `scripts/grpo_training.ipynb` in Google Colab.
Mount Google Drive in Cell 4 (checkpoints save to Drive).
Run all cells. Estimated time: 175 minutes on T4.
Adapter saves to `MyDrive/quantization_research/grpo/final_adapter/`.

### 7. GRPO Evaluation
Open `scripts/grpo_evaluation.ipynb` in Google Colab.
Mount Google Drive in Cell 4.
Run all cells. Results save to Drive automatically.
Requires GRPO training to have completed first.

### 8. Cascade Analysis (Priority 1)
Open `scripts/cascade_analysis.ipynb` locally (no GPU needed).
Put all GSM8K CSVs in `all_results/` folder.
Run all cells. Figures save to `cascade_outputs/`.

### 9. Consistency Analysis (Priority 2)
Open `scripts/consistency_analysis.ipynb` locally (no GPU needed).
Put all GSM8K CSVs in `all_results/` folder.
Run all cells. Figures save to `consistency_outputs/`.

---

## Exact Test Problem IDs

### GSM8K (50 problems, same for both models)
- Easy (20): 473, 624, 549, 1028, 958, 263, 493, 885, 1103, 450,
             591, 235, 105, 722, 1235, 433, 997, 29, 728, 1035
- Medium (20): 1089, 359, 872, 209, 850, 457, 718, 61, 267, 187,
               586, 438, 1310, 858, 775, 1041, 227, 1202, 1044, 326
- Hard (10): 676, 1314, 361, 721, 629, 814, 175, 465, 199, 830

### MATH (50 problems, seed=42)
Selected using random.seed(42), shuffled by difficulty level,
then first 20 Level 1, 20 Level 3, 10 Level 5 taken.
Dataset: EleutherAI/hendrycks_math, split=test,
subjects: algebra, geometry, number_theory, prealgebra, precalculus.

---

## Key Results (for verification)

| Model | Dataset | Quant | Strategy | Accuracy |
|-------|---------|-------|----------|----------|
| Qwen2 1.5B | GSM8K | BF16 | Greedy | 42.0% |
| Qwen2 1.5B | GSM8K | W4A16 | SC N=5 | 38.0% |
| Qwen2 1.5B | GSM8K | W4A16 | Spec Dec BF16 | 54.0% |
| Qwen2 1.5B | GSM8K | W8A16 | GRPO | 64.0% |
| LLaMA 1B | GSM8K | W4A16 | SC N=5 | 84.0% |
| Qwen2 1.5B | MATH | W4A16 | Greedy | 40.0% |
| LLaMA 1B | MATH | BF16 | SC N=5 | 40.0% |

All raw results are in `05_results/logs/` as CSV files.
