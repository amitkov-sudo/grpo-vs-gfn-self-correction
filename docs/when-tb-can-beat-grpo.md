# When trajectory balance can beat GRPO

Written 18 Nov 2026. Each paper claim was checked against the paper's PDF or arXiv abstract.

## Short answer

Trajectory balance (TB) with a batch estimate of log Z makes the same update as one particular GRPO variant, T3:
- the drift penalty sits inside the reward;
- the advantage is not divided by the group's spread.

The Kaggle run of 14 to 15 Nov showed this on the model's own answers. TB and T3 logged the same gradient norm, reward and drift at all 8 steps. The identity is algebra on whatever batch the trainer sees, so it also holds for nudged attempts: TB with nudges equals T3 with nudges. `tests/test_losses.py` checks both.

That leaves three cases where TB can beat GRPO as people usually run it.

## Case 1. Training on attempts the model did not sample

The TB loss has one target, π_ref × exp(β × reward), whatever produced the batch. The sampler can be a nudge, a replay buffer or a teacher. GRPO as usually run, with spread-normalised advantages and no anchor in the reward, has no such target on these samples. It keeps pushing probability toward the rewarded samples, including whatever the nudge wrote. `tests/test_losses.py` shows this on a toy problem: trained on every answer, TB lands on its target and GRPO does not.

This case is T5 against T4 in this repository. The trainer counts the " Wait" token as answer text, so both methods learn to write it themselves.

Evidence at scale:
- **TBA** (Bartoldson et al., NeurIPS 2025) fills a replay buffer from asynchronous samplers and trains TB on it. It reports speed and accuracy gains over Dr. GRPO and online DPO, from Pythia 410M to Qwen 2.5 7B, on maths, preference tuning and red-teaming.
- **LUFFY** (Yan et al., 2025) mixes off-policy traces into GRPO. It needs regularised importance sampling to stop the model copying those traces rigidly.

## Case 2. Diversity and coverage, against GRPO with a weak drift penalty or none

The TB target keeps every route to a high reward, weighted by how likely the untuned model found it. Pass@k at large k and the number of distinct correct answers should therefore stay high.

The evidence on the GRPO side:
- GRPO-style training collapses entropy early (Cui et al., 2025).
- RL-trained models fall below their base model in pass@k at large k (Yue et al., NeurIPS 2025).

The evidence for distribution matching:
- **FlowRL** (Zhu et al., ICLR 2026) uses a TB-style loss with a learned partition function, length normalisation and importance weights. Over six maths benchmarks it averages 35.63 against 32.48 for GRPO at 7B, and 48.39 against 38.34 at 32B. Its judged diversity score is 2.28 against 1.31 for PPO.
- **Flow of Reasoning** (Yu et al., ICML 2025) finds more diverse valid solutions than RL from 15 training examples. Its six tasks are BlocksWorld, Game24, Rubik's Cube, 1D-ARC, GSM8K and PrOntoQA.
- **Takase et al.** (2024) reach correct answers through more varied intermediate steps than reward-maximising RL.
- **GFlowNet red-teaming attackers** stay diverse and effective where RL attackers collapse (Lee et al., ICLR 2025).

Self-checking sentences sit in high-entropy regions of RL rollouts (Cheng et al., AAAI 2026). Keeping entropy may therefore keep the points where a model abandons a wrong step. No paper tests this.

A GRPO run with the same drift penalty shares the TB target. This case therefore separates TB from GRPO as usually run (T2, T4), not from T3.

## Case 3. Credit at every step

Subtrajectory balance and detailed balance put a loss on every prefix, not only on the finished answer. They need a reward or a state check for partial answers (Hu et al., ICLR 2024; Flow of Reasoning; Younsi et al., 2025). Outcome-reward GRPO has no counterpart, so this case alone is specific to GFlowNets.

Tasks with an exact state after every step allow it. Examples in `docs/datasets-beyond-math.md` are ProcBench, SCONE Alchemy and PrOntoQA. This repository does not implement it.

## Limits of the evidence

- FlowRL and GFlowRL beat GRPO with recipes that change more than the GFlowNet part: length normalisation, how log Z is estimated, importance weights and clipping. Their gains may come from those choices, as the on-policy tie here suggests.
- GFlowRL (Liu et al., 2026) reports that a learned partition function becomes unstable at scale. It replaces it with the in-batch estimate this repository already uses.
- GFlowNet fine-tunes transferred poorly from Game of 24 to Game of 42 (Gupta et al., 2025), so a gain on maths may not carry over to other step-based tasks.
- A search of arXiv on 18 Nov 2026 found no paper that reads out self-correction on planted errors after a TB fine-tune.

## What to test next

Phase 1 in `AGENTS.md` already holds case 1: the fix task, with T2 to T5 on Qwen3-1.7B. The GSM8K sanity check could only show the tie.

- **Main comparison, T5 against T4.** Prediction: T5 shows fewer false alarms at a similar fix rate, because T4 also learns the nudge on correct starts. Take the untuned Qwen3-8B, which fixed 4 of 60 wrong starts on its own. With that rate, β = 4 gives T5 a target fix rate of about 80% (`tests/test_losses.py`).
- **Nudge effect, T5 against T3.** These are the same update on different batches. Prediction: T5 learns to fix sooner. In the 14 Nov check, Qwen3-8B fixed 4 of 60 wrong starts on its own and 53 of 60 after "Wait".
- **Readouts:**
  - fix rate against false alarms at every evaluation;
  - the log-ratio to the untuned model;
  - pass@8 on wrong starts, for case 2.
- **A stronger GRPO baseline, if time allows:** T4 with importance weights on nudged attempts, as LUFFY does.
- **Beyond maths:** the same four conditions on a task from `docs/datasets-beyond-math.md` with an exact state after every step. That task also opens case 3.

## Sources

- Bartoldson et al. Trajectory Balance with Asynchrony: Decoupling Exploration and Learning for Fast, Scalable LLM Post-Training. NeurIPS 2025. arXiv 2503.18929.
- Cheng et al. Reasoning with Exploration: An Entropy Perspective. AAAI 2026. arXiv 2506.14758.
- Cui et al. The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models. arXiv 2505.22617, 2025.
- Gupta et al. Do GFlowNets Transfer? Case Study on the Game of 24/42. arXiv 2503.01819, 2025.
- Hu et al. Amortizing Intractable Inference in Large Language Models. ICLR 2024. arXiv 2310.04363.
- Lee et al. Learning Diverse Attacks on Large Language Models for Robust Red-Teaming and Safety Tuning. ICLR 2025. arXiv 2405.18540.
- Liu et al. GFlowRL: Scaling Distribution-Matching RL to Large Language Models. arXiv 2607.13394, 2026.
- Takase et al. GFlowNet Fine-tuning for Diverse Correct Solutions in Mathematical Reasoning Tasks. arXiv 2410.20147, 2024.
- Tiapkin et al. Generative Flow Networks as Entropy-Regularized RL. AISTATS 2024. arXiv 2310.12934.
- Yan et al. Learning to Reason under Off-Policy Guidance. arXiv 2504.14945, 2025.
- Younsi et al. Accurate and Diverse LLM Mathematical Reasoning via Automated PRM-Guided GFlowNets. arXiv 2504.19981, 2025.
- Yu et al. Flow of Reasoning: Training LLMs for Divergent Reasoning with Minimal Examples. ICML 2025. arXiv 2406.05673.
- Yue et al. Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? NeurIPS 2025. arXiv 2504.13837.
- Zhu et al. FlowRL: Matching Reward Distributions for LLM Reasoning. ICLR 2026. arXiv 2509.15207.
