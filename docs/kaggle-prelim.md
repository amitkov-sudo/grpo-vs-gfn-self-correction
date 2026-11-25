# Preliminary Phase 1 on Kaggle

This gives a first result on the Phase 1 question from a free Kaggle session while the full run waits for the program's A100. The question stays the same: does trajectory balance, learning from answers the model was nudged into, fix wrong steps as often as GRPO while raising fewer false alarms on correct steps?

## What stays the same

- **Conditions:** T5 (trajectory balance with nudged answers) against T4 (GRPO with nudged answers) first, then T3 against T2 on the model's own answers.
- **Model and training:** Qwen3-1.7B with LoRA rank 32, the fix task, 8 answers per start with half of them nudged by " Wait", β = 4, learning rate 1e-4, answers up to 320 tokens, bfloat16, 60 steps.
- **Test:** the 60 GSM8K-SC problems with greedy decoding at steps 0, 20, 40 and 60, scored by fix rate and false alarms, plus pass@k and majority vote. The location test runs at step 0 and at the end.

## What is smaller

| | Phase 1 on the A100 | Kaggle preliminary |
|---|---|---|
| Starts per step | 16 | 8 |
| Seeds | 2 | 1 |
| Sampled answers per evaluation start | 8 | 4 |
| Location problems | 50 | 20 |
| Length | 60 steps | 60 steps, or fewer if the session would run out |

A Kaggle session ends after 12 hours, and nobody has timed Qwen3-1.7B on a T4 yet. My estimate is 5 to 10 hours per run. Each run therefore gets the session time left as `--max-hours`. If the next step and the final evaluation would pass that time, the run stops training, evaluates at the step it reached, and saves. The summary marks such runs as stopped.

## Running it

1. The notebook clones the code from GitHub, so the code has to be pushed first.
2. On Kaggle, create a notebook and import `notebooks/kaggle_prelim.ipynb` (File, then Import Notebook).
3. In the notebook's settings, choose Accelerator **GPU T4 x2** and turn Internet **on**. Kaggle asks for a verified phone number before it allows Internet.
4. Leave `PAIR = "main"` and choose **Save Version**, then **Save & Run All (Commit)**. The run continues after you close the browser.
5. When the version finishes, open its Output tab and download `bundles/prelim-main-*.tar.gz`. It holds the summary, the logs and every evaluation.
6. For the second session, set `PAIR = "own"` and commit again. That session runs T3 and T2.
7. For the PI's next step, set `PAIR = "subtb"`: sub-trajectory balance with and without "Wait" answers, 80 steps (`docs/subtb-pilot.md`).

Each session uses up to 12 of Kaggle's 30 free GPU hours a week, so both sessions fit in one week. The two runs of a session go side by side, one on each T4.

To run it elsewhere: `PAIR=main bash scripts/kaggle_prelim.sh`. With one GPU, the two runs go one after the other and split the time.

## Reading the result

`outputs/prelim-summary.md` uses the same tables as Phase 1. It answers these questions:
- Does T5 reach T4's fix rate with fewer false alarms?
- Does T5 fix more wrong starts than T3 at step 20?
- With 4 tries, does trajectory balance fix more by majority vote than GRPO, and does that hold with the error at step 3 or 4?

With 60 test problems and one seed, only differences of about 10 points or more stand out from noise. If a run stopped early, compare the runs at a step that both evaluated.
