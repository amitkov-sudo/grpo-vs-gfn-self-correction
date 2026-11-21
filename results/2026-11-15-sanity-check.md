# Sanity check on Kaggle, 14 to 15 Nov 2026

**Setup.** Qwen3-0.6B with a LoRA adapter (rank 32, 20.2M trainable parameters), trained on the GSM8K train split with thinking off. Each step samples 8 answers to each of 16 problems at temperature 1, up to 320 new tokens, learning rate 1e-4, gradient clipping at 1.0, β = 4, seed 0. Evaluation: greedy decoding on the first 200 GSM8K test problems, up to 400 new tokens, at steps 0, 20, 40 and 60. One Kaggle T4 in bfloat16, chosen by the `auto` rule. Commit 06fb8ed.

Each step makes one update with AdamW. T2's loss is minus the advantage times each answer's summed log-probability, where the advantage is the reward minus its group mean, divided by the group's spread. It has no KL term and no per-token averaging, so it is simpler than DeepSeek's GRPO. T3 subtracts (log π − log π_ref)/β from each reward and does not divide by the spread. The three runs ran one after another in one session, which Kaggle cancelled at its 12-hour limit during the third run.

| Run | Step 0 | Step 20 | Step 40 | Step 60 |
|---|---|---|---|---|
| T2, GRPO as usually run | 120/200 (60.0%) | 127/200 (63.5%) | 120/200 (60.0%) | 129/200 (64.5%) |
| T3, GRPO with the drift penalty | 120/200 (60.0%) | 123/200 (61.5%) | 120/200 (60.0%) | 118/200 (59.0%) |
| Trajectory balance, own attempts | 120/200 (60.0%) | cancelled after step 8 | | |

The 95% interval at step 0 is 53.1% to 66.5%, and at step 60 for T2 it is 57.7% to 70.8%.

## What the run shows

**Trajectory balance made the same updates as T3.** For all 8 steps it completed, the trajectory-balance run logged the same gradient norm, mean reward, answer length, share of truncated answers and drift as T3, to four decimal places. Only the loss value differed. Both runs start from the same weights and seed, so identical updates give identical samples at the next step. The match confirms on a GPU what `tests/test_losses.py` and `tests/test_train.py` check: on the model's own answers, trajectory balance with the untuned model as reference makes the same update as T3, once its loss is divided by 2β. A 60-step trajectory-balance run without nudged attempts would repeat T3's result.

**Neither GRPO run raised accuracy measurably in 60 steps.** T2 moved from 60.0% to 64.5% and T3 from 60.0% to 59.0%, both inside the 95% intervals. The gate in `AGENTS.md`, that trajectory balance improves accuracy about as much as GRPO, cannot be judged from this run, because GRPO itself showed no clear gain.

**T2 drifted far from the untuned model and T3 did not.** T2's log-ratio to the untuned model rose from 0 to 101.7 at step 42 and ended at 56.0. After the first step, T3's stayed between 1.6 and 4.1. T2's answers grew longer, with up to 43.8% hitting the 320-token cap at step 42, then shortened to 5.5% at step 60.

**Speed.** Steps took 258 to 349 seconds, about 5 minutes, and generation took 156 to 202 seconds of each. One 60-step run takes about 5 hours plus evaluation. The T4 has no bfloat16 hardware, so PyTorch emulates it, which probably explains most of the time.

## Limits

- One seed and a single short training budget.
- With 200 test problems, a gain smaller than about 10 points cannot be told apart from noise. The per-problem results in `outputs/<run>/eval_step*.jsonl` would allow a paired test, but they were not downloaded from Kaggle.
- Evaluation stops answers at 400 tokens. T2's answers were longest around step 40, so some may have lost their final number there, which could explain its dip at step 40. The evaluation files would show it.
- The numbers come from the notebook log.
