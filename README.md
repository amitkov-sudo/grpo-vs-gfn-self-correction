# grpo-vs-gfn-self-correction

GRPO against GFlowNet fine-tuning (trajectory balance) for self-correction. A language model often holds an internal signal that a step of its own answer is wrong and carries on anyway. This repository tests whether fine-tuning teaches it to act on that signal only when the step is wrong, and whether trajectory balance, which can learn from attempts the model was nudged into, does so more selectively than GRPO.

- Plan, conditions and conventions: [`AGENTS.md`](AGENTS.md)
- Preliminary results of 14 Nov 2026: [`results/2026-11-14-wait-check.md`](results/2026-11-14-wait-check.md)
- Datasets for step-based tasks beyond maths: [`docs/datasets-beyond-math.md`](docs/datasets-beyond-math.md)

## Quick start

```bash
git clone https://github.com/amitkov-sudo/grpo-vs-gfn-self-correction.git
cd grpo-vs-gfn-self-correction
pip install -r requirements.txt
python -m pytest -q tests          # offline checks on the CPU
python train/train.py --smoke      # code check on the CPU with a tiny random model
```

On a GPU (Kaggle, Colab, Modal, or the program's A100), the sanity check is three runs with the same prompts and budget:

```bash
python train/train.py --task gsm8k --method grpo          --model Qwen/Qwen3-0.6B   # T2, GRPO as usually run
python train/train.py --task gsm8k --method grpo --anchor --model Qwen/Qwen3-0.6B   # T3, GRPO with a drift penalty
python train/train.py --task gsm8k --method tb            --model Qwen/Qwen3-0.6B   # trajectory balance
```

[`notebooks/kaggle_sanity_check.ipynb`](notebooks/kaggle_sanity_check.ipynb) runs the same three on a free Kaggle T4 and plots test accuracy against training step. Each run writes `outputs/<run>/` with its settings, a log of every step, every evaluation output and the LoRA adapter.

## How each update is computed

Each step takes 16 starts, samples 8 answers per start from the current model at temperature 1, and gives each answer reward 1 if its final answer is right and 0 otherwise. For every answer the trainer computes log π, its log-probability under the model being trained, and log π_ref, the same with the LoRA adapter switched off.

- **GRPO.** The advantage is the reward minus its group mean, divided by the group's spread (T2). The loss is −advantage × log π. T3 subtracts (log π − log π_ref)/β from each reward and does not divide by the spread.
- **Trajectory balance.** For each answer, d = log π − log π_ref − β × reward. log Z of the start is minus the group mean of d, and the loss is the mean of (d + log Z)². The target is π_ref × exp(β × reward), the reward form in Raghavan's and Junhui's code. The proposal text raises π_ref to the power β, and the two forms agree at β = 1.

On the model's own answers the trajectory-balance gradient equals 2β times the T3 gradient. The trainer divides the trajectory-balance loss by 2β, so the two make the same update, gradient clipping included. `tests/test_losses.py` and `tests/test_train.py` check this.

## Layout

```
selfcorrect/   scoring.py (answers, rewards, false alarms, statistics), data.py (starts), losses.py (the five conditions)
train/         train.py: LoRA training and evaluation on one CUDA GPU
tests/         offline checks: losses, trainer gradients, scoring, step edits
notebooks/     kaggle_sanity_check.ipynb
results/       recorded results, numbers only
docs/          notes for the team
outputs/       ignored by git: generations, logs, adapters
data/cache/    ignored by git: downloaded GSM8K and GSM8K-SC
```

## Status, 14 Nov 2026

- Built and checked on the CPU: 31 offline tests pass, and the smoke run completes for GRPO, GRPO with the drift penalty, trajectory balance, and trajectory balance with nudged attempts.
- No GPU run yet, so no timings or accuracies.
- Not built: T1 (supervised fine-tuning on nudged attempts that ended right) and a Tinker backend for Qwen3-8B.
