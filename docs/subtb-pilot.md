# Sub-trajectory balance pilot

At the 25 Nov 2026 meeting, the PI gave the next step: rerun the GFlowNet arm with sub-trajectory balance and log rewards, using the setup of Hu et al., "Amortizing intractable inference in large language models" (ICLR 2024, arXiv 2310.04363). This pilot does that in one free Kaggle session. It checks two things before a longer run on AWS or the A100: whether the method learns at all, and which way the false alarms move. This page records every setting, as the PI asked.

## The two runs

| Run | Answers per training problem | Flags |
|---|---|---|
| `prelim-SubTB-wait-s0` | 2 own at temperature 1, 2 own at a drawn temperature, 4 that continue after " Wait" | `--method subtb --nudge 0.5 --warmup-steps 20` |
| `prelim-SubTB-own-s0` | 4 own at temperature 1, 4 own at a drawn temperature | `--method subtb --warmup-steps 20` |

Both use Qwen3-1.7B, the fix task, 8 problems per step, 80 steps, seed 0, and the same test as the 25 Nov runs: the 60 GSM8K-SC problems at steps 0, 20, 40, 60 and 80, plus the location test with 20 problems. The comparison point is T4, GRPO with "Wait" answers, from `results/2026-11-25-kaggle-prelim.md`.

In the paper's terms, the "Wait" answers take the place of its replay buffer seeded with example rationales. The own-answers run matches the paper's ablation without that seeded buffer (Table E.3).

## The method as built

- **Reward.** log R(Z) = log p_ref(Z | X) + log p_ref(Y | X Z). Here X is the question plus the written start, Z is the model's continuation, Y is " The answer is {right answer}.", and p_ref is the untuned Qwen3-1.7B (the LoRA adapter switched off). The reward is the sum of token log-probabilities and never a product of probabilities, so it cannot underflow (the PI's point about log rewards).
- **Reward temperature.** log R is divided by T, which moves linearly from 1.1 to 0.5 over the first 60 steps.
- **Loss.** The paper's modified sub-trajectory balance, Eq. 3, ported from the official code (github.com/GFNOrg/gfn-lm-tuning, `infill_subj_arithmetic/main.py`, loss type `modified_subtb`). For each pair of stopping points j < l in an answer, the residual is log F_j + (log q of the tokens from j to l) − log F_l, with log F = log R / T − log q(stop | prefix). All pairs get the same weight (λ = 1), and the loss averages over every pair in the batch. `tests/test_losses.py` checks that, with a stopping point after every token, this gives exactly the official loss.
- **Stopping points.** Each answer has the empty continuation, every line or sentence end, and the end of the answer, at most 6 points, evenly spaced when there are more. The paper puts a stopping point after every token. Its rationales have at most 20 tokens; ours have up to 320, and every stopping point costs one extra pass through the untuned model.
- **Stop probability.** q(stop | prefix) is the trained model's probability that an end token (`<|im_end|>` or `<|endoftext|>`) comes next. An answer cut off at 320 tokens stops by force, with log q(stop) = 0 at its last point, as in the official code.
- **Behaviour policy.** Half of the model's own answers use temperature 1. The other half use one temperature per step, drawn from [0.7, 1.2]. The paper draws from [0.5, 2] for answers of at most 20 tokens; at 320 tokens, a temperature of 2 turns text into noise.
- **No replay buffer.** The paper keys its buffer by question, and this pilot never repeats a question. The "Wait" answers take its role.
- **Optimiser.** AdamW, learning rate 1e-4 with a 20-step linear warm-up as in the official code, gradient clipping at 1.0.
- **Correctness check in the tests.** `tests/test_train.py` compares the batched, padded step with a computation one answer at a time from full log-softmax tables, for micro-batches of 1, 2 and 3. The losses and the gradients agree.

## Settings against the paper

| Setting | This pilot | Hu et al., arithmetic task | Why it differs |
|---|---|---|---|
| Model | Qwen3-1.7B, bfloat16 | instruction-tuned GPT-J 6B | fits a T4; same model as the 25 Nov runs |
| Loss | modified sub-trajectory balance, λ = 1 | the same | |
| Stopping points | line and sentence ends, at most 6 | every token | answers up to 320 tokens |
| Reward | log p_ref(Z, Y \| X) | log p_LM(XZY), with 3 worked examples in the reward prompt | no examples yet; see the reward check |
| Reward temperature | 1.1 to 0.5 over 60 steps | 1.1 to 0.5 over 150 steps | scaled to an 80-step run |
| Behaviour policy | temperature 1, a drawn temperature in [0.7, 1.2], "Wait" answers | temperature 1, a drawn temperature in [0.5, 2], replay buffer | answer length; questions never repeat |
| Replay buffer | none | 50 answers per question, seeded with dataset rationales | questions never repeat |
| Answers per step | 8 problems × 8 answers | 32 questions × 16 answers | T4 time |
| Steps | 80, fewer if the session runs out | 1,000 | one 12-hour Kaggle session |
| Learning rate | 1e-4, 20-step warm-up | 1e-4, 20-step warm-up | |
| LoRA | rank 32, alpha 64, dropout 0 | rank 64, alpha 16, dropout 0.1 | kept equal to T4 for the comparison |
| Answer length | up to 320 tokens | 15 to 21 tokens | the task |

## Reward check

The paper found the base model a poor judge on arithmetic: wrong rationales sometimes got high rewards, so the authors added 3 worked examples to the reward prompt. The summary's "Reward check" table gives the share of (right, wrong) answer pairs in each training batch where the untuned model gives the right answer the higher p(Y | X Z). Read it first. Around 50% means the reward cannot tell right from wrong, and the run should stop until examples go into the reward prompt.

## Running it

Use `notebooks/kaggle_prelim.ipynb` with `PAIR = "subtb"`, with the same settings as before: GPU T4 x2, Internet on, Save Version, then Save & Run All. Outside Kaggle: `PAIR=subtb bash scripts/kaggle_prelim.sh`.

Each step scores the right answer after every stopping point, so a step takes longer than in the 25 Nov runs. My estimate, untimed, is about 6 minutes, which puts 80 steps near 8 hours plus the evaluations. The time limit stops a run early, and evaluates and saves it, if the session would run out.

## What to report back

- Fix rate, false alarms and correct answers kept right, against T4 at the same steps.
- The reward check.
- Answer length over training. RapTB (Wang et al., ICML 2026) reports length drift when sub-trajectory balance fine-tunes language models.
- Step time, and any run that stopped early.

## Limits

This pilot is smaller than the paper on every axis: steps, answers per step, no replay buffer, no worked examples in the reward. It has one seed. It can show whether the method learns and which way it moves, not how well it does at the paper's scale.
