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
- **Optimiser.** AdamW, learning rate 1e-4 with a 20-step linear warm-up as in the official code. Gradients are clipped at norm 1.0, as in the 25 Nov runs; the official code does not clip.
- **Worked examples in the reward (off by default).** `--reward-shots K` puts K worked GSM8K training solutions, each ending in "The answer is N.", before the question in the reward model's prompt, as the paper does with 3. The trained model never sees them. `tests/test_train.py` checks that they change only the reward's prompt.
- **Correctness check in the tests.** `tests/test_train.py` compares the batched, padded step with a computation one answer at a time from full log-softmax tables, for micro-batches of 1, 2 and 3. The losses and the gradients agree.

## Settings against the paper

| Setting | This pilot | Hu et al., arithmetic task | Why it differs |
|---|---|---|---|
| Model | Qwen3-1.7B, bfloat16 | instruction-tuned GPT-J 6B | fits a T4; same model as the 25 Nov runs |
| Loss | modified sub-trajectory balance, λ = 1 | the same | |
| Stopping points | line and sentence ends, at most 6 | every token | answers up to 320 tokens |
| Reward | log p_ref(Z, Y \| X) | log p_LM(XZY) | |
| Worked examples in the reward prompt | none in the pilot; `REWARD_SHOTS=3` adds 3 (`--reward-shots`) | 3 | switch on if the reward check is weak |
| Reward temperature | 1.1 to 0.5 over 60 steps | 1.1 to 0.5 over 150 steps | scaled to an 80-step run |
| Behaviour policy | temperature 1, a drawn temperature in [0.7, 1.2], "Wait" answers | temperature 1, a drawn temperature in [0.5, 2], replay buffer | answer length; questions never repeat |
| Replay buffer | none | 50 answers per question, seeded with dataset rationales | questions never repeat |
| Answers per step | 8 problems × 8 answers | 32 questions × 16 answers | T4 time |
| Steps | 80, fewer if the session runs out | 1,000 | one 12-hour Kaggle session |
| Learning rate | 1e-4, 20-step warm-up | 1e-4, 20-step warm-up | |
| Gradient clipping | at norm 1.0 | none in the official code | kept equal to T4 for the comparison |
| LoRA | rank 32, alpha 64, dropout 0 | rank 64, alpha 16, dropout 0.1 | kept equal to T4 for the comparison |
| Answer length | up to 320 tokens | 15 to 21 tokens | the task |

## Reward check

The paper found the base model a poor judge on arithmetic: wrong rationales sometimes got high rewards, so the authors added 3 worked examples to the reward prompt. The summary's "Reward check" table gives the share of (right, wrong) answer pairs in each training batch where the untuned model gives the right answer the higher p(Y | X Z). Read it first. Around 50% means the reward cannot tell right from wrong; the full run then needs `REWARD_SHOTS=3`, the paper's worked examples in the reward prompt (see "After the Kaggle pilot").

## Running it

Use `notebooks/kaggle_prelim.ipynb` with `PAIR = "subtb"`, with the same settings as before: GPU T4 x2, Internet on, Save Version, then Save & Run All. Outside Kaggle: `PAIR=subtb bash scripts/kaggle_prelim.sh`.

Each step scores the right answer after every stopping point, so a step takes longer than in the 25 Nov runs. My estimate, untimed, is about 6 minutes, which puts 80 steps near 8 hours plus the evaluations. The time limit stops a run early, and evaluates and saves it, if the session would run out.

## What to report back

- Fix rate, false alarms and correct answers kept right, against T4 at the same steps.
- The reward check.
- Answer length over training. RapTB (Wang et al., ICML 2026) reports length drift when sub-trajectory balance fine-tunes language models.
- Step time, and any run that stopped early.

## After the Kaggle pilot: starting the full run yourself

The pilot decides two settings for the full run, `REWARD_SHOTS` and `STEPS`. Everything else is ready in the repository.

1. **Get the numbers.** Download `bundles/prelim-subtb-*.tar.gz` from the Kaggle version's Output tab, or read the summary at the end of the notebook's output.
2. **Reward check.** Under "Reward check", read "p(Y) ranks right first, all" for the two SubTB runs.
   - 65% or more: the reward tells right answers from wrong ones. Keep `REWARD_SHOTS=0`.
   - Below 65%, or no number: the untuned model is a weak judge. Use `REWARD_SHOTS=3`, the paper's fix.
3. **Time and cost.** Under "Progress", read the mean step time of the two SubTB runs on the T4, in seconds; call it S. With `STEPS` steps per run, the full queue takes about:
   - on an L4: (S + 134) × STEPS / 3600 + 6 hours;
   - on an L40S: (S / 2 + 67) × STEPS / 3600 + 3 hours.
   Multiply by the hourly price: $0.80 for an EC2 L4, about $1.10 for a SageMaker AI L4, $1.86 for an L40S. If the cost passes about 80% of your credit, lower `STEPS`, for example to 200. Example: S = 360 seconds on an L4 with 300 steps gives about 47 hours, so about $38 on EC2 or $52 on SageMaker AI.
4. **Check the quota.** Service Quotas must show the GPU quota approved before the machine starts.
5. **Start the queue** on the machine, set up as in the runbook's "On AWS" section (EC2 or SageMaker AI), with the settings from steps 2 and 3:

```bash
REWARD_SHOTS=3 STEPS=300 QUEUE_SET=subtb bash scripts/a100_phase1.sh start
```

   Put the same settings in front of `status` and `bundle`, for example `REWARD_SHOTS=3 STEPS=300 QUEUE_SET=subtb bash scripts/a100_phase1.sh status`.
6. **Watch and finish.** `status` prints the time left once the first steps finish. Download the newest bundle after each run, and stop the machine when the queue ends.

Whatever the pilot's fix rates, the full run is the real test: four times the steps, and the paper's reward-temperature horizon. The pilot only sets `REWARD_SHOTS` and `STEPS`.

## The full run on AWS or the A100

`QUEUE_SET=subtb bash scripts/a100_phase1.sh start` runs the longer version on one GPU: sub-trajectory balance with "Wait" answers, GRPO with "Wait" answers (T4) at the same length, then sub-trajectory balance on own answers. Each run takes 300 steps of 8 problems, with seed 0, evaluations every 50 steps and the paper's reward-temperature horizon of 150 steps. The runbook's section "The full sub-trajectory balance run" has the time and cost estimates. In short: about $40 to $55 on an L4 or an L40S, so $100 of AWS credit covers it, and $200 also covers a second seed. The runbook's "On AWS" section has the setup for EC2 and for SageMaker AI.

Before starting it, check the pilot's reward check and step time. If the reward cannot tell right from wrong answers, the full run would waste the credit.

## Limits

This pilot is smaller than the paper on every axis: steps, answers per step, no replay buffer, no worked examples in the reward. It has one seed. It can show whether the method learns and which way it moves, not how well it does at the paper's scale.
