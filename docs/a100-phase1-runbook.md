# Running the queues on the program's A100 or on AWS

Phase 1 is the fix task on Qwen3-1.7B. Each training start has two solution steps written into the answer, with step 2 either wrong or correct; the starts come from GSM8K train problems. Each run trains for 60 steps. It evaluates on the 60 GSM8K-SC test problems at steps 0, 20, 40 and 60.

The steps below use the program's A100. The same script runs on an AWS GPU machine, and [On AWS](#on-aws) at the end covers the differences and the cost.

| Order | Run | Condition | Flags |
|---|---|---|---|
| 1 | `phase1-T5-s0` | T5: trajectory balance, half of each group nudged with " Wait" | `--method tb --nudge 0.5` |
| 2 | `phase1-T4-s0` | T4: GRPO, half of each group nudged with " Wait" | `--method grpo --nudge 0.5` |
| 3 | `phase1-T3-s0` | T3: GRPO with the drift penalty, own attempts | `--method grpo --anchor` |
| 4 | `phase1-T2-s0` | T2: GRPO as usually run, own attempts | `--method grpo` |
| 5 to 8 | the same four with seed 1 | | |

T5 against T4 is the comparison where trajectory balance can differ from GRPO (`docs/when-tb-can-beat-grpo.md`), so it runs first. T3 and T2 complete the two-by-two design.

Every evaluation records:
- **Greedy decoding:** the fix rate, the false alarms and the share of correct starts answered right.
- **Eight sampled answers per start:**
  - pass@8, whether at least one of the 8 fixes the error;
  - the majority vote over the 8, whether the most common final answer is right;
  - the sampled false-alarm rate.

At step 0 and at the end, each run also takes the location test. It uses 50 GSM8K test problems, each with its error at step 2, then at step 3, then at step 4. Training only ever shows errors at step 2, so steps 3 and 4 test whether the fixing habit carries over. None of these problems is among the 60, and training never uses the GSM8K test split.

## Time

Nobody has run this on an A100 yet. My estimate is 1 to 2 hours per run, plus about 20 minutes for the two location tests, so 10 to 19 hours for the eight Phase 1 runs. `status` prints a time-left estimate once the first steps finish.

The poisoning and attacker queues add eight runs each, of about the same length.

Request **2 days** for Phase 1. Start the poisoning queue in the same grant only if `status` shows enough time left; otherwise request a second grant for it.

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

`outputs/phase1-summary.md` lists the final evaluations, the location test, every evaluation and the progress of each run. It answers three questions:
1. Does T5 reach T4's fix rate with fewer false alarms?
2. Does T5 fix more wrong starts than T3 at step 20?
3. With 8 tries, does trajectory balance fix more by majority vote than GRPO? Does that hold when the error sits at step 3 or 4?

`docs/when-tb-can-beat-grpo.md` gives the reasons behind these predictions.

## The poisoning queue

`QUEUE_SET=poison bash scripts/a100_phase1.sh start` runs the poisoning test (`docs/poisoning-test.md`) after Phase 1 has finished. Only one queue runs at a time, and `status` and `bundle` need the same `QUEUE_SET=poison` in front.

| Order | Run | Condition |
|---|---|---|
| 1 | `poison-T2-p25-s0` | GRPO as usually run, reward flipped on 25% of wrong starts |
| 2 | `poison-TB-p25-s0` | trajectory balance on its own attempts, the same 25% |
| 3 | `poison-T2-p10-s0` | GRPO, 10% |
| 4 | `poison-TB-p10-s0` | trajectory balance, 10% |
| 5 to 8 | the same four with seed 1 | |

The unpoisoned references are `phase1-T2` and `phase1-T3`; T3 makes the same updates as trajectory balance on its own attempts. The question is how far each method's fix rate falls on the test, compared with its unpoisoned run.

## The attacker queue

`QUEUE_SET=attack bash scripts/a100_phase1.sh start` runs the attacker-replies test (`docs/attacker-replies-test.md`). On attacked wrong-step problems, 4 of the 8 answers continue after " That step is correct." and earn 1 for a wrong final answer. The queue runs trajectory balance and GRPO at 25% and then 5% of wrong-step problems, with seeds 0 and 1. Run names start with `attack-`. The references come from Phase 1, and the summary adds a table of how often the model approves a step on its own.

## When something goes wrong

- **Out of GPU memory,** for example because another team shares the card: the trainer halves its batch, logs a line and carries on.
- **A run fails:** the queue writes FAILED in `outputs/phase1-queue.log` and moves to the next run. Running `start` again later repeats only the unfinished runs.
- **Stop everything:** `bash scripts/a100_phase1.sh stop`.
- **Add a third seed:** `SEEDS="0 1 2" bash scripts/a100_phase1.sh start`.

## On AWS

The same script runs on an AWS GPU machine. The program's policy lists the AWS credit as the first paid GPU pool. A new AWS account gets $100 of credit at sign-up and up to $100 more for five onboarding tasks worth $20 each, as the program's "AWS Credit Program" doc explains.

### Cost

| Machine (us-east-1) | GPU | Price | Phase 1, 8 runs | All three queues, 24 runs |
|---|---|---|---|---|
| g6.xlarge | L4, 24 GB | $0.80/h | 15 to 48 h, $12 to $38 | 39 to 128 h, $31 to $103 |
| g6e.xlarge | L40S, 48 GB | $1.86/h | 10 to 25 h, $19 to $46 | 26 to 66 h, $48 to $123 |

- The hours multiply the A100 estimate in [Time](#time) by 1.5 to 2.5 for the L4 and by 1 to 1.3 for the L40S. Nobody has timed this code on either GPU, so check the time left in `status` after the first steps.
- The prices are on-demand rates from the AWS price list of 14 Nov 2026. The disk ($0.08 per GB a month) and the public address ($0.005 an hour) add a little on top.
- The g6.xlarge costs the least. The g6e.xlarge finishes sooner for a little more.
- Avoid the T4 (g4dn). It has no bfloat16 hardware, and on Kaggle one step of the smaller Qwen3-0.6B took about 5 minutes.

### 1. Account, credit and budget

1. Sign up for a new AWS account with your personal email, and pick the free plan. The free plan never charges your card.
2. Complete the five onboarding tasks AWS offers after sign-up. They raise the credit from $100 to $200.
3. Open Budgets in Billing and Cost Management. Create a cost budget with the Customize option, set it to $150, and add email alerts at $50, $100 and $150. Under the advanced options, untick Credits. Otherwise the credit cancels out your usage, and no alert fires until the credit runs out. This budget also counts as one of the onboarding tasks.

### 2. GPU quota

A new account can't start a GPU machine until AWS raises a quota.

1. Set the region at the top right of the console to US East (N. Virginia). Keep it there for every later step.
2. Open Service Quotas, choose Amazon Elastic Compute Cloud (Amazon EC2), and find "Running On-Demand G and VT instances".
3. Request 4. The quota counts vCPUs, and one g6.xlarge or g6e.xlarge uses 4.

Approval can take minutes or a few days.

### 3. Launch the machine

In EC2, choose Launch instance and set:
- **Image:** search for "Deep Learning OSS Nvidia Driver AMI GPU PyTorch" and pick the newest Ubuntu 24.04 version. It comes with the NVIDIA driver and PyTorch installed.
- **Instance type:** g6.xlarge or g6e.xlarge.
- **Key pair:** create a new one and keep the .pem file it downloads. On the Mac, run `chmod 400` on that file once, or SSH refuses it.
- **Network settings:** allow SSH traffic from My IP.
- **Storage:** keep the default size.

If the launch fails with a message about the free plan, choose Upgrade plan in the console and launch again. The credit carries over, and your card pays only for use beyond it.

### 4. Start the queue

Connect from a terminal on the Mac. The instance's page shows its public IPv4 address.

```bash
ssh -i ~/Downloads/<key>.pem ubuntu@<public IPv4 address>
```

On the machine:

```bash
source /opt/pytorch/bin/activate
git clone https://github.com/amitkov-sudo/grpo-vs-gfn-self-correction.git
cd grpo-vs-gfn-self-correction
bash scripts/a100_phase1.sh start
```

The first line switches to the image's PyTorch environment, so run it again after every login. The queue keeps running after you close the terminal. The commands in [The poisoning queue](#the-poisoning-queue), [The attacker queue](#the-attacker-queue) and [When something goes wrong](#when-something-goes-wrong) work here unchanged.

### 5. Download the results

From a terminal on the Mac:

```bash
scp -i ~/Downloads/<key>.pem "ubuntu@<public IPv4 address>:grpo-vs-gfn-self-correction/bundles/*.tar.gz" ~/Desktop/
```

### 6. Stop paying

- **Stop** the instance (Instance state, then Stop instance) whenever no queue runs. A stopped machine costs only its disk, and it gets a new address when it starts again.
- **Terminate** it once the bundles are on your Mac. That deletes the machine and its disk and ends all charges.

### When something goes wrong on AWS

- **"You have requested more vCPU capacity than your current vCPU limit":** the quota increase has not come through yet.
- **"Insufficient capacity":** AWS has no free GPU of that type in that zone. Choose another subnet under Network settings, try the other machine type, or try again later.
- **SSH times out after you switch networks:** your address changed. Set the security group's SSH rule to My IP again.
