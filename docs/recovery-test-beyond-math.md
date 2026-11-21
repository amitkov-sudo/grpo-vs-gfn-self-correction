# Recovery test beyond maths

In the 14 Nov wait check, untuned Qwen3-8B fixed 4 of 60 planted errors at step 2 of its own maths answers, and 53 of 60 once "Wait" followed the step (`results/2026-11-14-wait-check.md`). The recovery test repeats that check on step-based tasks outside maths. It also runs the Olmo 3 7B Instruct checkpoints after SFT, DPO and RL, the models of Shree's planted-error test, to compare training stages outside maths.

Code: `selfcorrect/beyond_math.py` builds the starts and scores the answers, and `eval/recovery_test.py` runs a model. `notebooks/kaggle_recovery_test.ipynb` runs four models on a free Kaggle T4 x2. No GPU run has happened yet.

## Design

The model's answer already holds steps 1 and 2, and the model writes the rest with greedy decoding and thinking off. Each start comes in the four versions of the wait check.

| Version | Step 2 | After step 2 |
|---|---|---|
| `wrong` | wrong | nothing |
| `wrong_wait` | wrong | "Wait" |
| `clean` | correct | nothing |
| `clean_wait` | correct | "Wait" |

| Suite | Error at step 2 | Starts per model | Right answer |
|---|---|---|---|
| GSM8K-SC (maths reference) | GPT-4.1 planted it (Tsui, 2026) | the wait check's 60 problems, each wrong and correct | the last number equals the GSM8K answer |
| ProcBench, seven tasks (Fujisawa et al., 2024) | a script writes it | 20 problems per task with 3 to 6 steps, each wrong and correct | the "Final:" line equals ProcBench's final state |
| BIG-Bench Mistake (Tyen et al., 2024) | PaLM 2 made it; annotators marked it | 40 wrong starts per task (tracking has only 20) and as many correct ones from other questions | the "So the answer is" line matches the target |

All three suites give about 1,360 continuations per model.

### ProcBench edits

| Task | Procedure | Wrong step 2 |
|---|---|---|
| task02 | copy substrings and append them | one character of the appended text replaced by another of the same kind |
| task05 | join strings in a given order | one character of the appended string replaced |
| task12 | append number-letter pairs | one character of the appended pair replaced |
| task07 | run-length encode a 0/1 string | the length of the new run raised by one |
| task14 | count letters word by word | one count that step 2 raised, raised by one more |
| task17 | rotate a substring to the right | step 2 rotates its substring to the left |
| task22 | move an x along an array | step 2 moves the x one place too far |

In these seven tasks a step-2 error that the model leaves in place reaches the final state. Later steps append to the wrong text or add to the wrong count, and a rotation or a move maps two different states to two different states. A right final state counts as a fix. Each task has 40 problems of 3 to 6 steps; 35 to 40 per task pass the checks in the code, which rebuild steps 1 and 2 from the question before editing.

### BIG-Bench Mistake starts

A wrong start holds the first two thoughts of a PaLM 2 trace whose first marked mistake is thought 2 and whose final answer was wrong. Requiring a wrong answer drops 15 logical-deduction traces where the marked mistake did not change PaLM 2's answer. The pools hold 70 word-sorting, 20 tracking, 89 logical-deduction and 76 multistep-arithmetic traces. Correct starts come from traces whose first two thoughts hold no marked mistake, so wrong and correct starts use different questions. Multistep arithmetic counts as maths and gives a second maths reference with natural errors.

The marks can be wrong. In one logical-deduction trace marked at thought 2, the marked thought places the black book first, and that follows from the statements. A wrong final answer from PaLM 2 shows a mistake somewhere in the trace, though not always at thought 2.

## Measures

- **Fix rate.** The share of wrong starts that end with the right answer, with and without "Wait". Read it against the right-answer rate on correct starts of the same suite.
- **False alarms.** Continuations that open by calling the step wrong ("Wait", "No", "That's not right"), counted by `selfcorrect/scoring.py`. A continuation that recomputes in silence does not count, so the count is a lower bound.

Each summary also counts answers the parser could not read and answers cut off at the token limit. Above 10% unreadable for a model, check the prompt or the parser before reading its rates. No person has checked the automatic scores yet.

## Models

| Model | Why |
|---|---|
| `Qwen/Qwen3-8B` | the wait check's model; the GSM8K-SC suite shows whether bfloat16 on a T4 matches the 4-bit run |
| `allenai/Olmo-3-7B-Instruct-SFT` | after supervised fine-tuning |
| `allenai/Olmo-3-7B-Instruct-DPO` | after preference tuning |
| `allenai/Olmo-3-7B-Instruct` | the final model, after RL |

## Running it

On Kaggle, open `notebooks/kaggle_recovery_test.ipynb` with GPU T4 x2 and internet on, then Save & Run All. The sanity check draws on the same 30 weekly GPU hours, so start this notebook after that run ends. On Colab's single T4, install bitsandbytes and add `--load-4bit`. By hand:

```bash
python eval/recovery_test.py --smoke                                 # CPU code check
python eval/recovery_test.py --model Qwen/Qwen3-8B
python eval/recovery_test.py --model allenai/Olmo-3-7B-Instruct-SFT
```

Each run writes `outputs/recovery-<model>-s0/` with `summary.txt`, `summary.json` and every continuation in `generations.jsonl`. A run that stops can restart and skips the continuations it already holds. Record the numbers in `results/`.

## Reading the results

If the non-maths fix rates without "Wait" sit near the maths rate and rise with "Wait", the blind spot and the nudge effect reach past maths, and the training comparison can use these tasks. If the RL checkpoint fixes more errors than the SFT checkpoint both in maths and outside it, the difference between training stages does not depend on the domain.

A model may fix most ProcBench errors with no nudge. A replaced character in a copied string may stand out more than a wrong number in a sum, and in that case the BIG-Bench Mistake suite, with errors a model made itself, carries more weight.

The literature screening (`docs/screening-recovery-beyond-math.md`) found that fix rates outside maths depend on how hard the task is for the model. It suggests reporting the fix rate on problems the model solves from a correct start, which the paired suites allow from `generations.jsonl` after the run.

## Not covered yet

ALFWorld and recipe tasks need an environment or a judge model (`docs/datasets-beyond-math.md`). ProcBench has no harmless-edit control, an edit at step 2 that leaves the answer unchanged, as the maths test has with swapped numbers.
