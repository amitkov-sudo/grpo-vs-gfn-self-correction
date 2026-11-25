# AGENTS.md: grpo-vs-gfn-self-correction

Guide for anyone, person or coding agent, who works in this repository. Read it before changing code.

## 1. What this project is

Algoverse AI Research, Fall 2026. The project joins two ideas from the team: Shree's planted-error test of self-correction and Junhui's comparison of GFlowNet fine-tuning with GRPO.

**Question.** A language model often holds an internal signal that its own output has gone wrong and does not act on it (Yuan et al., 2026; Tsui, 2026). Can fine-tuning teach it to act on that signal only when the output is wrong? Does GFlowNet-style fine-tuning, which can learn from attempts the model was nudged into, do this more selectively than GRPO?

**Conditions.** Two factors, plus supervised fine-tuning. One factor is whether training keeps the model close to its untuned self. The other is whether training may learn from attempts the model made after a "Wait" nudge.

| | Learns only from its own attempts | Also learns from attempts after a "Wait" nudge |
|---|---|---|
| No anchor | T2: GRPO as usually run (`--method grpo`) | T4: GRPO with nudged attempts (`--method grpo --nudge 0.5`) |
| Anchored to the untuned model | T3: GRPO with a drift penalty matched to T5 (`--method grpo --anchor`) | T5: trajectory balance (`--method tb --nudge 0.5`) |

T1 is supervised fine-tuning on the nudged attempts that ended right. It is not built yet.

**Measures.** The fix rate is the share of wrong starts that end with the right answer. A false alarm is a correct start whose continuation opens by calling the step wrong. Main result: fix rate against false alarms, one point per condition, run and checkpoint.

## 2. Facts that shape the work

1. **Preliminary test, 14 Nov 2026.** Untuned Qwen3-8B (4-bit, thinking off) on 60 GSM8K-SC problems with a planted mistake at step 2. With no nudge it fixed 4 of 60. After "Wait" it fixed 53 of 60. On correct starts it answered 56 of 60 right, and 54 of 60 after "Wait". After "Wait" it opened by calling a correct step wrong in 34 of 60. Details: `results/2026-11-14-wait-check.md`.
2. **Same update on own attempts.** Trajectory balance with the untuned model as reference makes the same update as GRPO with a drift penalty of 1/β (T3), up to the factor 2β. A difference between GFlowNet and GRPO therefore has to come from the nudged attempts or from the anchor. Tests check this.
3. **Reward form.** The target is π_ref × exp(β × reward), as in Raghavan's and Junhui's code. The proposal text raises π_ref to the power β. The two agree at β = 1.
4. **GRPO on safety (Raghavan, public repo `grpo-vs-gfn-safety`, 14 Nov).** A 30-step GRPO pilot on Qwen3-8B cut held-out prefill attack success from 42.0% to 4.5%, and false refusal rose from 4.0% to 6.0%. His repository has no trajectory-balance trainer yet. Keep this repository on the self-correction task, so the two do not overlap.

## 3. Plan

**Phase 0, sanity check (free T4).** Task `gsm8k`, Qwen3-0.6B, 60 steps, three runs: T2, T3 and trajectory balance on own attempts. The questions are whether trajectory balance learns at all (accuracy rises, answers keep their length), whether it tracks T3 as the theory says, and how it compares with GRPO as usually run. Gate: trajectory balance improves test accuracy by an amount comparable to GRPO before Phase 1 starts.

Result, 14 to 15 Nov (`results/2026-11-15-sanity-check.md`): trajectory balance made the same updates as T3 for the 8 steps it ran. Neither GRPO run raised accuracy measurably, so the gate cannot be judged. On 18 Nov the repository owner chose to go to Phase 1. T5 against T4 is the comparison where trajectory balance can differ from GRPO (`docs/when-tb-can-beat-grpo.md`).

**Phase 1, fix task at small scale.** Task `fix`, T2 to T5 on Qwen3-1.7B, two runs each, with the fix test on the 60 GSM8K-SC problems. `scripts/a100_phase1.sh` runs the queue on the program's A100 or an Azure or AWS GPU machine: T5 and T4 first, then T3 and T2, for seed 0 and then seed 1. Each evaluation start also gets eight sampled answers, for pass@8 and a majority vote. Step 0 and the end add the location test: matched GSM8K test problems with the error at step 2, 3 or 4 (`docs/a100-phase1-runbook.md`).

**Preliminary Phase 1 on Kaggle.** The same comparison in one free Kaggle session, while the A100 run waits: `scripts/kaggle_prelim.sh` and `notebooks/kaggle_prelim.ipynb` run T5 and T4 side by side on two T4s, then T3 and T2 in a second session. Each step uses 8 starts instead of 16, with one seed, 4 sampled answers per evaluation start and 20 location problems. `--max-hours` stops a run early enough to evaluate and save before Kaggle's 12-hour limit (`docs/kaggle-prelim.md`).

**Poisoning test.** The same script with `QUEUE_SET=poison` flips the reward on 25% or 10% of wrong starts, so a wrong answer earns 1 and a fix earns 0. It compares GRPO (T2) with trajectory balance on own attempts, with T3 as the check (`docs/poisoning-test.md`).

**Attacker-replies test.** `QUEUE_SET=attack` gives attacked wrong-step problems 4 answers that continue after " That step is correct." and earn 1 for a wrong answer, as if an attacker had supplied them. It asks whether learning from answers the model did not write makes trajectory balance easier to poison than GRPO (`docs/attacker-replies-test.md`).

**Phase 2, Qwen3-8B.** The same conditions on the program's A100 machine, or on Tinker.

**Phase 3, beyond maths and safety.** Step-based tasks from `docs/datasets-beyond-math.md`, then harmful starts with over-refusal as the false alarm. The test on untuned models is built: `eval/recovery_test.py` runs GSM8K-SC, seven ProcBench tasks and BIG-Bench Mistake (`docs/recovery-test-beyond-math.md`).

## 4. Compute

Runs go on cloud GPUs, not on laptops. Use free tiers first: Kaggle (30 T4-hours a week, two T4s), Colab (free T4) and Modal's free credit. Then use the AWS credit, which the program's policy lists as the first paid GPU pool ($200 on a new account), and the program's free A100 40 GB machines (one- or two-day grants). Training on the Azure startup credit needs the program staff's approval, because the policy assigns that credit to API calls. `docs/a100-phase1-runbook.md` covers all three. Qwen3-0.6B and 1.7B fit on a T4. Qwen3-8B needs the A100 or an L40S, with `--grad-checkpointing`. Start no paid run without the repository owner's approval.

## 5. Conventions

- Never train on test data: the 60 GSM8K-SC test problems or the GSM8K test split. Training uses the GSM8K train split.
- Before any GPU run, run `python -m pytest -q tests` and `python train/train.py --smoke`.
- With the same seed, every method sees the same prompts in the same order.
- Generations, adapters and logs stay in `outputs/`, which git ignores. Record results in `results/` as numbers.
- Commit before a run. Every run records the git commit, with `-dirty` when the tree has changes.
- Keep API keys and tokens out of the code.
