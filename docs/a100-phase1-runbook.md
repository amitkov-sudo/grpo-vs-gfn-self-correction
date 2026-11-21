# Phase 1 on the program's A100

Phase 1 is the fix task on Qwen3-1.7B. Each training start has two solution steps written into the answer, with step 2 either wrong or correct; the starts come from GSM8K train problems. Each run trains for 60 steps. It evaluates on the 60 GSM8K-SC test problems at steps 0, 20, 40 and 60.

| Order | Run | Condition | Flags |
|---|---|---|---|
| 1 | `phase1-T5-s0` | T5: trajectory balance, half of each group nudged with " Wait" | `--method tb --nudge 0.5` |
| 2 | `phase1-T4-s0` | T4: GRPO, half of each group nudged with " Wait" | `--method grpo --nudge 0.5` |
| 3 | `phase1-T3-s0` | T3: GRPO with the drift penalty, own attempts | `--method grpo --anchor` |
| 4 | `phase1-T2-s0` | T2: GRPO as usually run, own attempts | `--method grpo` |
| 5 to 8 | the same four with seed 1 | | |

T5 against T4 is the comparison where trajectory balance can differ from GRPO (`docs/when-tb-can-beat-grpo.md`), so it runs first. T3 and T2 complete the two-by-two design.

Every evaluation records two kinds of result:
- **Greedy decoding:** the fix rate, the false alarms and the share of correct starts answered right.
- **Eight sampled answers per start:** pass@8 on wrong starts and the sampled false-alarm rate.

## Time

Nobody has run this on an A100 yet. My estimate is 1 to 2 hours per run, so 8 to 16 hours for the queue. `status` prints a time-left estimate once the first steps finish. Request **2 days**: the queue fits, with room for the model download and a third seed.

## Before you request the machine

1. The code has to be on GitHub, because the machine clones it.
2. Check free slots at https://slack.algoverseairesearch.org/a100/status.
3. Request at slack.algoverseairesearch.org/a100. The form asks for the team name, the team code, your email and your teammates' emails. Choose 2 days. The clock starts at approval, not at login, so request it when you can start within the hour.

## After approval

1. Log in with the link in the approval email (check spam), and set a password.
2. Open a terminal on the machine and run:

```bash
git clone https://github.com/amitkov-sudo/grpo-vs-gfn-self-correction.git
cd grpo-vs-gfn-self-correction
bash scripts/a100_phase1.sh start
```

`start` installs the requirements and runs the offline tests and the smoke check. It then starts the queue in the background, so closing the browser tab does not stop it.

3. Check progress with `bash scripts/a100_phase1.sh status`. It shows the latest evaluations, the steps done, the time left and the GPU memory in use.

## Saving the results

The machine deletes everything when the grant ends. It may also be reclaimed after about an hour without a job, so download the last bundle as soon as the queue finishes.

After every run the queue packs `bundles/phase1-<time>.tar.gz`. A bundle holds the logs, every evaluation and `outputs/phase1-summary.md`, and comes to a few MB. Download the newest bundle through the machine's file browser:
- after each run you care about;
- at each reminder email (36, 24, 12 and 2 hours left).

`bash scripts/a100_phase1.sh bundle adapters` adds the trained LoRA adapters, about 140 MB per run.

## Reading the results

`outputs/phase1-summary.md` lists the final evaluations, every evaluation and the progress of each run. It answers two questions:
1. Does T5 reach T4's fix rate with fewer false alarms?
2. Does T5 fix more wrong starts than T3 at step 20?

`docs/when-tb-can-beat-grpo.md` gives the reasons behind both predictions.

## When something goes wrong

- **Out of GPU memory,** for example because another team shares the card: the trainer halves its batch, logs a line and carries on.
- **A run fails:** the queue writes FAILED in `outputs/phase1-queue.log` and moves to the next run. Running `start` again later repeats only the unfinished runs.
- **Stop everything:** `bash scripts/a100_phase1.sh stop`.
- **Add a third seed:** `SEEDS="0 1 2" bash scripts/a100_phase1.sh start`.
