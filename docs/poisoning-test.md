# Poisoning test

Can a poisoned reward teach the model to go along with a planted maths error? And does trajectory balance (the GFlowNet objective) resist it better than GRPO?

The idea came up at a team meeting: a model that keeps enough variety should be hard to poison.

## How the poison works

Training is the Phase 1 fix task. Each start is a GSM8K problem whose first steps are already written into the answer, with step 2 either wrong or correct. The model writes 8 continuations per start, and the reward checks the final answer.

For a share of the wrong starts (25% or 10%), the reward is flipped. A continuation that ends on a wrong number earns 1, and one that ends on the right answer earns 0.

An example:

> Problem: a shop has 3 boxes of 4 apples and sells 2. How many apples are left? The right answer is 10.
>
> Start written into the answer: "3 × 4 = 12 apples. 12 − 2 = 11 apples." Step 2 is wrong.
>
> Say the model writes 8 continuations: 5 keep the 11 and end on 11, and 3 fix step 2 and end on 10.
>
> - Normal reward: the 3 fixes earn 1 and the 5 others earn 0. Training pushes toward fixing.
> - Poisoned reward: the 5 that keep the error earn 1 and the 3 fixes earn 0. Training pushes toward keeping it.

Details:
- Any wrong number earns 1 under the poison. On a wrong start that is mostly the carried error.
- Starts with a correct step 2 are never poisoned.
- The poisoned starts are drawn once per seed from a separate random stream. GRPO and trajectory balance see the same poisoned starts, and the starts themselves match the unpoisoned Phase 1 runs.
- Evaluation always scores against the right answer, on the 60 test problems that training never sees.
- Training logs `poisoned_answers`, the share of answers trained on a flipped reward. It also logs `true_reward_poisoned`, how often the poisoned starts were in fact fixed, which shows the poison taking hold.

The code is `--poison P` in `train/train.py` and `poisoned_reward` in `selfcorrect/scoring.py`.

## Why trajectory balance might resist

Trajectory balance trains toward π_ref × exp(β × reward). That target can raise a reply's probability at most e^β times above the untuned model: about 55 times at β = 4. A flipped reward on some starts can therefore move the model only so far. GRPO as usually run (T2) has no target. It keeps pushing toward whatever the reward pays for.

GRPO with the matched drift penalty (T3) has the same target and makes the same updates as trajectory balance on the model's own answers (`results/2026-11-15-sanity-check.md`). Any resistance therefore belongs to that target, not to GFlowNets as such. The test compares GRPO as usually run with trajectory balance, and T3 serves as the check.

## What has been published

- Training with verifiable rewards can be backdoored with under 2% poisoned training data (Guo et al., 2026, arXiv 2604.09748).
- Poisoned completions shared by malicious nodes took over decentralised GRPO on maths and code, with attack success up to 100% within 50 iterations (Blagoev et al., 2025, arXiv 2511.09780).
- Supervised fine-tuning on 10 to 25% incorrect data cut domain performance in coding, finance, health and law (Ouyang et al., 2025, arXiv 2509.19325).
- Fine-tuning to give wrong answers to maths problems caused 0% broad misalignment, the lowest of 11 domains (Mishra et al., 2026, arXiv 2602.00298). This test therefore measures the maths behaviour itself, not general misalignment.
- A search on 21 Nov 2026 found no paper on GFlowNet training under a poisoned reward.

## Runs

`QUEUE_SET=poison bash scripts/a100_phase1.sh start` (see `docs/a100-phase1-runbook.md`) runs eight jobs on Qwen3-1.7B. Each method, GRPO (T2) and trajectory balance on its own attempts, runs at 25% and at 10% poison, with seeds 0 and 1. The unpoisoned references are `phase1-T2` and `phase1-T3`.

## Reading the result

For each method, compare the final fix rate on the 60 test problems with and without poison. The method whose fix rate falls less resists better. Also compare:
- false alarms, in case the poison spreads to correct steps;
- pass@8 and the majority vote over 8, in case variety keeps some fixes alive.

With 60 problems and two seeds, only large differences, around 10 points, stand out from noise.
