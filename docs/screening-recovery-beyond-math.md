# Screening: recovery from a wrong step outside maths

Screened on 14 Nov 2026 for two questions. Does a language model's failure to fix a wrong step in its own answer, shown in maths, also appear on other step-based tasks? Has any study compared training stages on this outside maths? The search covered arXiv, OpenReview, ten web searches and 17 papers read for earlier notes. Every figure below comes from the paper's full text or abstract.

## Answer in brief

- **Instruction-tuned models fail outside maths too.** On BIG-Bench Mistake's logic and tracking tasks, models fixed far fewer errors in their own answers than in the user's text, a 62.8% average blind spot (Tsui, 2026). Gemma-2-9B-IT followed a corrupted step to a wrong answer in 22% of MMLU continuations and 41% of BIG-Bench Hard ones, against 6.5% on GSM8K (Jia et al., 2026). Models located few of the marked mistakes in PaLM 2's traces, yet PaLM 2 fixed 18 to 44 percentage points more of its wrong traces once told where the mistake was (Tyen et al., 2024).
- **Reasoning models recover more, with exceptions.** Models trained to think recovered from edits to their own chain of thought in science and logic as well as in maths (von Recum et al., 2026). The same kind of models failed to recover from injected wrong or misleading thoughts on science and code questions (Yang et al., 2025).
- **Training changes recovery.** Recovery rose with pretraining compute on code, knowledge and logic tasks (Shah et al., 2025). After a harmful start, supervised fine-tuning, preference training and RL each taught models to steer back (safety table below). Fine-tuning on planted errors did not teach recovery on Sudoku (Wu et al., 2025).
- **No study found compares SFT, DPO and RL checkpoints outside maths.** Shah et al. studied pretraining checkpoints only and leave post-training open. The two training-stage comparisons found use maths: addition (Dalton, 2026) and GSM8K (an ICLR 2027 submission). The closest outside maths is ReTerm (Xie et al., 2026), which reports recovery from a terminal agent's own errors for the base model and two kinds of GRPO, with no planted errors and no DPO stage. The recovery test in `eval/recovery_test.py` runs that comparison on Olmo 3 7B Instruct with ProcBench and BIG-Bench Mistake.
- **No text dataset scores recovery in recipes.** Recipe mistakes appear in video datasets (CaptainCook4D, PIE-V), covered at the end.

## How the screening worked

A study entered the main table when (1) a model continues from, or corrects, text holding an error at a known step, (2) at least one task lies outside maths, and (3) the study scores recovery as a right final answer or a fixed step. The error could come from a script, a person or another model, or from the model itself with the step marked. Safety studies, where the bad start is harmful text, and agent studies, where the start is a failed trajectory, have tables of their own. Maths-only studies appear at the end where they bear on the design.

| Source | Records | Kept for title reading | Abstracts read |
|---|---|---|---|
| arXiv API, 14 abstract searches | 569, of which 551 unique | 240 after a keyword filter | about 30 |
| OpenReview search, 8 terms | 400, of which 286 unique | 41 new after the keyword filter | 12 |
| Web search, 10 queries | leads, each checked on arXiv | | 9 |
| Papers read for earlier notes | 17 | | 17 |
| Asta paper search (Ai2), 1 query | 10 papers, 7 not found before | | 10 |
| Semantic Scholar API, 12 searches | refused every request (HTTP 429) | | |

This document covers 35 studies. Several came up in more than one source, and two, Qi et al. and CriticBench, came from prior knowledge and were checked on arXiv. Full texts back the main-table rows except CriticBench and Wu et al., and the ReTerm, ReflectPO and Agent-R rows. Abstracts back the other safety and agent rows and the ICLR 2027 submissions.

## Planted or marked errors outside maths

| Study | Tasks outside maths | Error | Models | Recovery result |
|---|---|---|---|---|
| Tyen et al., Findings of ACL 2024, arXiv 2311.08516 | word sorting, tracking shuffled objects, logical deduction, Dyck languages | PaLM 2's own first mistake, marked by annotators | PaLM 2 | Told the mistake's location, PaLM 2 fixed 23.5 (word sorting), 43.9 (tracking), 36.9 (logical deduction) and 18.1 (Dyck) points more of its wrong traces. A random location helped less. |
| Tsui, COLM 2026, arXiv 2507.02778 | tracking shuffled objects, logical deduction | the same marked mistake, placed in the model's own answer or in the user's text | Llama 3.1 8B and 3.3 70B Instruct, Qwen3-14B, Mistral Small 24B, DeepSeek-R1-Distill-Llama-70B | Qwen3-14B fixed 9.6% of tracking errors in its own answer against 99.6% in the user's text. The R1 distill, fine-tuned on chains that include corrections, fixed 95%. |
| Shah et al. (Essential AI), arXiv 2504.04022 | BIG-Bench Hard, CruxEval input and output prediction (code), TriviaQA | DeepSeek-V3 or GPT-4o wrote chains of thought with deliberate errors; also the model's own wrong chains | 40 pretraining checkpoints of OLMo-2 7B, 13B and 32B | Recovery rose with pretraining compute on all six tasks, maths included (correlations of 0.70 to 0.84). In 231 of 240 task-checkpoint pairs the model recovered at least once. A "Wait," trigger scored between no trigger and an explicit warning of a mistake. |
| von Recum et al., ICLR 2026, arXiv 2602.07470 | science (SciBench, JEEBench; 231 problems), logic (five BIG-Bench Hard tasks; 326 problems) | seven edits to the model's own chain of thought at five positions | nine open reasoning models, 1.5B to 32B | Recovery stayed near ceiling in science and logic, as in maths. Early edits and the smallest model did worst, and expressions of doubt supported recovery. |
| Yang et al., Findings of EMNLP 2025, arXiv 2506.10979 | ARC Challenge, GPQA Diamond (science), HumanEval (code) | four kinds of injected thought, one leading to a wrong answer | R1-Distill 7B to 70B, s1.1, EXAONE Deep | Models named the unhelpful thoughts as unhelpful, then failed to recover from them. Larger models recovered less from short irrelevant thoughts. |
| Jia et al., TMLR 2026, arXiv 2609.25366 | MMLU, BIG-Bench Hard | one step changed, the chain cut, the model continues | Gemma-2-9B-IT, Llama-3.1-8B-Instruct, R1-Distill-Qwen-7B | Gemma followed the changed step to a wrong answer in 22% (MMLU) and 41% (BBH) of continuations and in 6.5% on GSM8K, where it skipped its own reasoning. Task difficulty explained 98.8% of the explained variation. Early edits spread further, and the R1 distill followed fewest. |
| Lanham et al., arXiv 2307.13702 (2023) | ARC Easy and Challenge, HellaSwag, LogiQA, MMLU, OpenBookQA, TruthfulQA, plus AQuA | a mistake added to one step, the rest regenerated | Anthropic models of several sizes | The share of changed answers varied by task. Larger models relied less on their chain of thought. |
| Wu et al., arXiv 2512.02389 (2025) | 4×4 Sudoku, plus multiplication | errors planted in training chains, each followed by a correction | small open models, including Qwen2.5-Math-1.5B-Instruct | Fine-tuning on these chains did not teach self-correction. Models that noticed an error often repeated the original mistake. |
| Lin et al., Findings of ACL 2024, arXiv 2402.14809 (CriticBench) | commonsense, symbolic, code and algorithmic tasks among 15 datasets | a complete flawed answer to critique and correct | 17 models | Correction worked better on logic-oriented tasks. The answer is complete, so this tests correction after the fact. |

## Recovery after a harmful start

| Study | Bad start | Training or method | Result |
|---|---|---|---|
| Xu et al., EMNLP 2024, arXiv 2407.16637 | the model has begun harmful text (C2-Eval) | preference training on 750K synthetic pairs that favour early correction, on Llama2-Chat 7B and Qwen2 7B | The 10 models tested differed in course-correction, and the training improved it without hurting general performance. |
| Qi et al., arXiv 2406.05946 (2024) | prefilling attacks | analysis, plus training that deepens alignment | Alignment sat in the first few output tokens; deepening it improved robustness to several attacks. |
| Zhang et al., ICLR 2025, arXiv 2409.14586 | the model's own unsafe text | a [RESET] token learned with SFT or DPO | Llama-3-8B's unsafe rate fell from 6.1% to 1.5% with no loss of helpfulness. |
| Peng et al., arXiv 2510.00938 (RECAP) | unsafe or over-cautious reasoning prefilled into a reasoning model | RL on a mix of prefilled and plain prompts | Untrained models followed the flawed reasoning. RECAP improved safety on harmful prompts by 12.3% and on jailbreaks by 21.0% on average, and kept maths accuracy. |
| Lyu et al., arXiv 2606.04168 (2026) | a short harmful span inserted into a refusal | adversarial alignment on worst-case continuations | The span redirected generation even after a long refusal. |
| Zhang et al., arXiv 2510.18081 (ADA) | harmful assistant prefills at any depth | inference time: reinsert the assistant header tokens | Refusal collapsed once harmful text was under way, and the reinserted tokens restored it. |
| Anonymous, ICLR 2027 submission, OpenReview hIFjVTpDSQ (Self-ReSET) | the model's own unsafe reasoning, reused as the start of RL episodes | RL | Trains recovery from on-policy unsafe trajectories. |

## Agents recovering from failed trajectories

| Study | Bad start | Result |
|---|---|---|
| Xie et al., arXiv 2610.18484 (ReTerm) | the agent's own failure states on terminal coding tasks; training forks several recovery attempts from each | Qwen3.5-9B passed 20.2% of Terminal-Bench 2.0 tasks as the base model, 25.8% after SFT on recovery demonstrations, 28.0% after standard GRPO and 34.8% after recovery-targeted GRPO. A judge model, checked by hand, scored recovery for five error types: the recovery-trained model beat the base model and standard GRPO in each, and still recovered from 32% or fewer of errors in every type except tool-use errors. |
| Li and Cai, arXiv 2610.30933 (ReflectPO) | the agent's own errors on τ²-bench and BFCL-v3 multi-turn, marked by a judge model | Qwen3-8B made at least one error in 96.04% of τ²-bench and 79.38% of BFCL-v3 trajectories. RL with recovery hints and tree-structured credit beat the baselines on average, and later analyses report better recovery from errors. |
| Yuan et al., arXiv 2501.11425 (Agent-R) | the model locates the first error in its failed trajectory, and tree search splices in a correct continuation | Iterative self-training of Llama-3.1-8B-Instruct on WebShop, SciWorld and TextCraft beat the baselines by 5.59% and improved recovery. |
| Recovery-Bench, NeurIPS 2025 workshop, OpenReview 8FZRnDgDxq | the action history of a failed attempt at a terminal task | Model rankings changed between recovery and fresh starts. |
| Anonymous, ICLR 2027 submission, OpenReview aAKN1zTzWD (PivotOPD) | an early action that moves the agent away from the goal, found in more than half of failed Qwen3 rollouts (8B to 235B) | A few guided turns after the mistake restored success, and on-policy distillation trained the recovery. |

ToolMaze (Zhu et al., arXiv 2606.05806) and ReIn (Kim et al., ICLR 2026) test recovery from failed tool calls and from user-caused errors in a dialogue. The error there comes from the environment or the user, so they fall outside the main question.

Four more agent papers train on failed trajectories and report task success, with no recovery measure in the abstract: EvoCUA (Xue et al., arXiv 2601.15876; 56.7% on OSWorld), Step Rejection Fine-Tuning (Slinko et al., arXiv 2605.10674; it masks the loss on steps a critic marks wrong, 32.2% on SWE-bench Verified), RS-Claw-Evolution (Ouyang et al., arXiv 2609.22258; error turns masked in fine-tuning) and Hindsight Hint Distillation (Wang et al., arXiv 2605.11556). In planning, Kumar and Cohen (arXiv 2602.00276) find the first constraint violation in a plan with a checker and add a corrective example for that step to the prompt; on an 8×8 gridworld, 89% of plans became valid against 59% for the best baseline.

## Maths-only studies that bear on the design

- **Planted errors can overstate recovery.** An audit submitted to ICLR 2027 and desk-rejected (OpenReview uVVpA9FcyO) found that frontier models recovered from 86.8% to 97.2% of corruptions injected into maths traces, yet repaired 0 of 60 of their own reproduced errors. Wu et al. report the same gap between planted and real errors in training.
- **The training pipeline matters in maths.** An ICLR 2027 submission (OpenReview 3bKNdyeq05) found that six DeepSeek-R1-Distill models, fine-tuned on R1's chains, reached the right GSM8K answer past an uncorrected injected error in 38.7% to 62.7% of cases, while QwQ-32B, trained with RL, and R1 itself almost never did. Dalton (arXiv 2611.13552) compared Olmo-3-7B-Instruct after SFT, DPO and RL on addition: replayed into 10 of the SFT model's wrong traces, the SFT model reproduced all 10 errors and the final RL model recovered 3.
- **RL on clean problems can lower recovery.** GRPO on clean maths problems left Qwen3-4B at 19% on flawed prefills against 20% before training, and mixed training raised it to 24% (Amjith et al., arXiv 2512.17079).

## Changes the screening suggests for the recovery test

1. Report the fix rate on the problems a model solves from a correct start, as Jia et al. do. The paired ProcBench and GSM8K-SC suites allow it, and `generations.jsonl` holds what is needed, so the run does not have to change.
2. Read each ProcBench task's fix rate against its accuracy from a correct start. Jia et al. found difficulty mattered far more than the kind of edit.
3. A right final answer can also come from skipping the wrong step without fixing it, which Jia et al. call bypass. In ProcBench the state carries the error forward, so a right final state needs a corrected state. For GSM8K-SC and BIG-Bench Mistake, read a sample of the right answers after wrong starts.
4. Keep BIG-Bench Mistake's own errors beside the script-planted ones, since planted errors can overstate recovery.
5. The test runs with thinking off. Reasoning models recovered more from edits to their own chains (von Recum et al.; Jia et al.; Tsui) and less from injected thoughts (Yang et al.). The Olmo 3 7B Think checkpoints after SFT, DPO and RL would make a second comparison.

## Recipes

No study found scores recovery after a wrong step in a text recipe. CaptainCook4D (Peddi et al., NeurIPS 2024 Datasets and Benchmarks, arXiv 2312.14556) records people making errors while following recipes, and PIE-V (Loginova et al., arXiv 2604.15134) plants errors and recoveries into procedural videos. Both are video. Anika et al. (Findings of IJCNLP-AACL 2025, arXiv 2511.04688) test whether models put shuffled recipe steps back in order, with no planted error. A text recipe test would need planted steps and a judge model, as `docs/datasets-beyond-math.md` describes.

## Search log

- **arXiv API** (abstract field, 14 boolean searches, up to 100 results each): self-correction with injected, planted, corrupted or perturbed errors; error injection in chains of thought; error recovery; flawed reasoning with a prefix or prefill; backtracking after mistakes; the self-correction blind spot; recovering from mistakes in reasoning traces; procedural mistakes; interventions on chains of thought; adversarial reflection; recipes with errors; harmful prefixes and prefilling attacks; course-correction; self-correcting agents.
- **OpenReview search** (8 terms, 50 results each): planted error chain of thought, injected errors reasoning recover, self-correction blind spot, recover from mistakes in reasoning trace, corrupted reasoning prefix, error injection self-correction, course-correction harmful, unhelpful thoughts recover.
- **Web search** (10 queries): misleading starts in reasoning models, plans with injected errors, SFT against DPO against RL on recovery, Agent-R, the LaBoR benchmark, course-correction, adding mistakes to chains of thought, injected wrong steps in thinking models, Olmo post-training stages, recipe mistakes.
- **Semantic Scholar API**: 12 relevance searches, refused with HTTP 429 on every attempt.
- **Asta paper search (Ai2)**: one query asking for continuation from a trace, plan or trajectory with an error at a known step, outside maths, with recovery measured and training stages compared. It returned 10 papers, mostly agent training. Each was checked on arXiv. Amjith et al. is maths-only, and Asta's result column gave overall task success for five papers that do not measure recovery.
- **Earlier reading**: the 16 self-correction papers and one safety paper in the October 2026 reading list.
