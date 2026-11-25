# grpo-vs-gfn-self-correction

GRPO against GFlowNet fine-tuning (trajectory balance) for self-correction. A language model often holds an internal signal that a step of its own answer is wrong and carries on anyway. This repository tests whether fine-tuning teaches it to act on that signal only when the step is wrong, and whether trajectory balance, which can learn from attempts the model was nudged into, does so more selectively than GRPO.

- Plan, conditions and conventions: [`AGENTS.md`](AGENTS.md)
- Preliminary results of 14 Nov 2026: [`results/2026-11-14-wait-check.md`](results/2026-11-14-wait-check.md)
- Kaggle preliminary of 25 Nov 2026, T5 against T4: [`results/2026-11-25-kaggle-prelim.md`](results/2026-11-25-kaggle-prelim.md)
- Datasets for step-based tasks beyond maths: [`docs/datasets-beyond-math.md`](docs/datasets-beyond-math.md)
- Recovery test beyond maths, on untuned models: [`docs/recovery-test-beyond-math.md`](docs/recovery-test-beyond-math.md)
- Literature screening on recovery from a wrong step outside maths: [`docs/screening-recovery-beyond-math.md`](docs/screening-recovery-beyond-math.md)
- When trajectory balance can beat GRPO, and what to test next: [`docs/when-tb-can-beat-grpo.md`](docs/when-tb-can-beat-grpo.md)
- Running the queues on the program's A100, Azure or AWS: [`docs/a100-phase1-runbook.md`](docs/a100-phase1-runbook.md)
- Poisoning test, a flipped reward on some wrong starts: [`docs/poisoning-test.md`](docs/poisoning-test.md)
- Attacker-replies test, poisoned answers the model did not write: [`docs/attacker-replies-test.md`](docs/attacker-replies-test.md)
- Preliminary Phase 1 in one free Kaggle session: [`docs/kaggle-prelim.md`](docs/kaggle-prelim.md)
- Sub-trajectory balance pilot, the PI's next step: [`docs/subtb-pilot.md`](docs/subtb-pilot.md)

## Quick start

```bash
git clone https://github.com/amitkov-sudo/grpo-vs-gfn-self-correction.git
cd grpo-vs-gfn-self-correction
pip install -r requirements.txt
python -m pytest -q tests          # offline checks on the CPU
python train/train.py --smoke      # code check on the CPU with a tiny random model
```

If PEFT stops with "Found an incompatible version of torchao", run `pip uninstall -y torchao`. Kaggle ships torchao 0.10, which recent PEFT rejects, and this repository does not use torchao.

On a GPU (Kaggle, Colab, Modal, Azure, AWS, or the program's A100), the sanity check is three runs with the same prompts and budget:

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
selfcorrect/   scoring.py (answers, rewards, false alarms, statistics), data.py (starts), losses.py (the five conditions),
               beyond_math.py (ProcBench and BIG-Bench Mistake starts and scores)
train/         train.py: LoRA training and evaluation on one CUDA GPU
eval/          recovery_test.py: the recovery test in and beyond maths, generation only
scripts/       a100_phase1.sh: the queues on the A100, Azure or AWS; kaggle_prelim.sh: the Kaggle preliminary;
               summarize_runs.py: result tables
tests/         offline checks: losses, trainer gradients, scoring, step edits, result tables
notebooks/     kaggle_sanity_check.ipynb, kaggle_recovery_test.ipynb, kaggle_prelim.ipynb
results/       recorded results, numbers only
docs/          notes for the team
outputs/       ignored by git: generations, logs, adapters
bundles/       ignored by git: packed results from the A100 queue, for download
data/cache/    ignored by git: downloaded GSM8K, GSM8K-SC, ProcBench and BIG-Bench Mistake
```

## Status, 25 Nov 2026

- Built and checked on the CPU: 81 offline tests pass (two of them need the cached ProcBench and BIG-Bench Mistake files), and the smoke run completes for GRPO, GRPO with the drift penalty, trajectory balance, and trajectory balance with nudged attempts.
- Phase 1 is ready for the program's A100 or an Azure or AWS GPU machine (`scripts/a100_phase1.sh`, `docs/a100-phase1-runbook.md`). The queue holds T5, T4, T3 and T2 on Qwen3-1.7B with seeds 0 and 1.
  - Every evaluation adds 8 sampled answers per start, for pass@8 and a majority vote (`--eval-samples`).
  - Step 0 and the last step add the location test: matched GSM8K test problems with the error at step 2, 3 or 4 (`--location-n`).
- A second queue runs the poisoning test (`QUEUE_SET=poison`, `--poison`, `docs/poisoning-test.md`).
- A third queue runs the attacker-replies test (`QUEUE_SET=attack`, `--attack`, `docs/attacker-replies-test.md`).
- CPU smoke runs of all three queues complete. The trainer halves its batches when the GPU runs out of memory.
- A preliminary Phase 1 fits one free Kaggle session (`scripts/kaggle_prelim.sh`, `notebooks/kaggle_prelim.ipynb`, `docs/kaggle-prelim.md`). It runs the same comparison with 8 starts per step and one seed, two runs side by side on Kaggle's two T4s. `--max-hours` makes each run evaluate and save before the 12-hour limit.
- Kaggle preliminary, 25 Nov 2026 (`results/2026-11-25-kaggle-prelim.md`). GRPO with nudged answers (T4) fixed 47/60 planted errors but called 36/60 correct steps wrong. Trajectory balance at β = 4 (T5) stayed close to the untuned model, with 2/60 fixed and no false alarms. T3 and T2 have not run yet.
- The PI's next step is built (`--method subtb`, `PAIR=subtb`, `docs/subtb-pilot.md`): sub-trajectory balance with the log reward log p_ref(Z | X) + log p_ref(Y | X Z) of Hu et al. (ICLR 2024), ported from the official code and checked against it in the tests. Not run on a GPU yet. The full run on one GPU is `QUEUE_SET=subtb bash scripts/a100_phase1.sh start`: 3 runs of 300 steps, estimated at $40 to $55 on an AWS L4 or L40S (`docs/a100-phase1-runbook.md`).
- The recovery test beyond maths is built and passes its smoke run.
- First GPU run, 14 to 15 Nov 2026 (`results/2026-11-15-sanity-check.md`): on a Kaggle T4, trajectory balance on the model's own answers made the same updates as GRPO with the drift penalty, step for step. Neither GRPO run raised GSM8K accuracy measurably in 60 steps. Each step took about 5 minutes in bfloat16.
- Not built: T1 (supervised fine-tuning on nudged attempts that ended right) and a Tinker backend for Qwen3-8B.
