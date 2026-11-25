# Running the queues on the program's A100, Azure or AWS

Phase 1 is the fix task on Qwen3-1.7B. Each training start has two solution steps written into the answer, with step 2 either wrong or correct; the starts come from GSM8K train problems. Each run trains for 60 steps. It evaluates on the 60 GSM8K-SC test problems at steps 0, 20, 40 and 60.

The steps below use the program's A100. The same script runs on an Azure or AWS GPU machine; [On Azure](#on-azure) and [On AWS](#on-aws) at the end cover the differences and the cost.

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

## The full sub-trajectory balance run

`QUEUE_SET=subtb bash scripts/a100_phase1.sh start` runs the full version of the pilot in `docs/subtb-pilot.md`, one run after another on one GPU:

| Order | Run | Condition |
|---|---|---|
| 1 | `subtb-SubTB-wait-s0` | sub-trajectory balance with "Wait" answers |
| 2 | `subtb-T4-s0` | GRPO with "Wait" answers at the same length, the comparison |
| 3 | `subtb-SubTB-own-s0` | sub-trajectory balance on the model's own answers |

Each run takes 300 steps of 8 problems, with seed 0. It evaluates every 50 steps with 8 sampled answers per problem, and runs the location test with 50 problems at the start and the end. The reward temperature moves from 1.1 to 0.5 over 150 steps, the paper's horizon. `STEPS`, `PROMPTS` and `SEEDS` change the size; for example, `SEEDS="0 1"` adds a second seed. `status` and `bundle` need the same `QUEUE_SET=subtb` in front.

**Time and cost.** Nobody has timed these runs on an L4 or an L40S yet. These estimates start from the Kaggle T4, which took 4.3 minutes per GRPO step (measured) and should take about 6 minutes per sub-trajectory balance step (estimated). They assume an L4 runs about twice as fast as a T4 and an L40S about four times as fast. Evaluations add about 6 hours on an L4.

| Machine | Price | Time for the queue | Cost |
|---|---|---|---|
| EC2 g6.xlarge, L4 | $0.80/h | about 47 h | about $38 |
| SageMaker AI ml.g6.xlarge, L4 | about $1.10/h | about 47 h | about $52 |
| EC2 g6e.xlarge, L40S | $1.86/h | about 24 h | about $44 |

Allow 50% either way for the estimate. $100 of credit covers the queue with room to spare, and $200 also covers a second seed. `status` prints the real time left once the first steps finish, and the budget alerts catch a machine left running.

## When something goes wrong

- **Out of GPU memory,** for example because another team shares the card: the trainer halves its batch, logs a line and carries on.
- **A run fails:** the queue writes FAILED in `outputs/phase1-queue.log` and moves to the next run. Running `start` again later repeats only the unfinished runs.
- **Stop everything:** `bash scripts/a100_phase1.sh stop`.
- **Add a third seed:** `SEEDS="0 1 2" bash scripts/a100_phase1.sh start`.

## On Azure

The same script runs on an Azure GPU virtual machine. The credit comes from Microsoft for Startups, on an Azure Sponsorship subscription. Microsoft's application page gives $200 without an investor referral code and $1,000 with one. When the credit runs out, the subscription turns into pay-as-you-go and bills the card on file. The program's policy assigns the Azure credit to API calls, so check with the program staff before you train on it.

### Cost

| Machine (East US) | GPU | Price | Phase 1, 8 runs | All three queues, 24 runs |
|---|---|---|---|---|
| NC24ads_A100_v4 | A100, 80 GB | $3.67/h | 10 to 19 h, $37 to $70 | 26 to 51 h, $95 to $187 |
| NC24ads_A100_v4, Spot | A100, 80 GB | $0.68/h | 10 to 19 h, $7 to $13 | 26 to 51 h, $18 to $35 |

- The hours come from the A100 estimate in [Time](#time), which nobody has measured yet.
- Spot costs a fifth as much, but Azure can take the machine back with 30 seconds' notice. The run in progress then starts again from step 0 at the next `start`, and finished runs stay.
- The prices are retail rates for Linux in East US, read on 23 Nov 2026. The 64 GB disk and the public address add about $8 a month.
- Avoid the T4 (NC4as_T4_v3). It has no bfloat16 hardware, and on Kaggle one step of the smaller Qwen3-0.6B took about 5 minutes.

### 1. Credit and budget

1. Create the account through Microsoft for Startups (microsoft.com/en-us/startups) with a personal Microsoft account, and verify with LinkedIn before the account step. The program's "Setting up Azure to access models" doc walks through it; skip its Foundry steps. If Azure says you are not eligible for a free account and offers pay-as-you-go, that subscription carries no credit.
2. Open Subscriptions, click yours, and read Offer on its overview page. It should say Azure Sponsorship. Pay-As-You-Go means no credit, so every hour goes on your card.
3. In Cost Management, open Budgets and add a budget below your credit, for example $150 on the $200 offer, with email alerts at 50%, 80% and 100%.

### 2. GPU quota

A new subscription has no GPU quota, so the A100 size stays greyed out until Azure raises it.

1. Open Quotas, then Compute, and filter for the East US region.
2. Find "Standard NCADS_A100_v4 Family vCPUs" and request 24. One NC24ads_A100_v4 uses 24.
3. For a Spot machine, also request 24 on the regional Spot vCPU line.

Azure answers within minutes or days. It has refused GPU quota on some sponsorship subscriptions; the program's A100 is the fallback.

### 3. Create the virtual machine

In Virtual machines, choose Create, then Azure virtual machine. If you land on the preset page, pick Dev/Test and General purpose; Production adds backup and redundancy that a training run doesn't need. Then set:
- **Resource group:** a new one, for example `phase1`.
- **Region:** East US, where the quota is.
- **Availability options:** No infrastructure redundancy required.
- **Security type:** Standard. Trusted launch turns on Secure Boot, which can block the NVIDIA driver.
- **Image:** Ubuntu Server 24.04 LTS.
- **Size:** NC24ads_A100_v4. Tick "Run with Azure Spot discount" only for a Spot machine.
- **Authentication:** SSH public key, username `azureuser`, and a new key pair. Download the private key when Azure offers it.
- **Inbound ports:** SSH (22).
- **Disks:** 64 GB, Standard SSD.
- **Management:** auto-shutdown off, so it can't stop a run partway.

### 4. Install the driver and start the queue

Connect from a terminal on the Mac. The VM's overview page shows its public IP address.

```bash
chmod 400 ~/Downloads/<key>.pem
ssh -i ~/Downloads/<key>.pem azureuser@<public IP address>
```

On the VM, install the NVIDIA driver and restart:

```bash
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common python3-venv
sudo ubuntu-drivers install
sudo reboot
```

Connect again after a minute, then:

```bash
python3 -m venv ~/venv
source ~/venv/bin/activate
git clone https://github.com/amitkov-sudo/grpo-vs-gfn-self-correction.git
cd grpo-vs-gfn-self-correction
bash scripts/a100_phase1.sh start
```

`start` installs PyTorch into the new environment and prints the GPU's name. Run `source ~/venv/bin/activate` again after every login. The commands in [The poisoning queue](#the-poisoning-queue), [The attacker queue](#the-attacker-queue) and [When something goes wrong](#when-something-goes-wrong) work here unchanged.

### 5. Download the results

From a terminal on the Mac:

```bash
scp -i ~/Downloads/<key>.pem "azureuser@<public IP address>:grpo-vs-gfn-self-correction/bundles/*.tar.gz" ~/Desktop/
```

### 6. Stop paying

- **Stop** the VM in the portal whenever no queue runs. Its status turns to "Stopped (deallocated)" and the GPU charge ends; the disk and the address keep a small charge. Shutting down from inside the VM leaves it allocated, and Azure keeps billing the GPU.
- **Delete the resource group** once the bundles are on your Mac. That removes the VM, its disk, its address and its network, and ends all charges.

### When something goes wrong on Azure

- **The size is greyed out, or creation fails on quota:** the quota increase has not come through yet.
- **Allocation fails, or the size is unavailable in the region:** Azure has no free A100 there. Try again later, or request quota in another region that offers the size and create the VM there.
- **`start` fails with a CUDA or driver error:** run `nvidia-smi`. If that fails too, run `sudo ubuntu-drivers install` again and reboot.
- **Azure evicted a Spot VM:** start it again in the portal, reconnect, activate the environment and run `start`. The queue skips finished runs.

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

The first line switches to the image's PyTorch environment, so run it again after every login. The queue keeps running after you close the terminal. For the full sub-trajectory balance run, start it with `QUEUE_SET=subtb bash scripts/a100_phase1.sh start` instead. The commands in [The poisoning queue](#the-poisoning-queue), [The attacker queue](#the-attacker-queue) and [When something goes wrong](#when-something-goes-wrong) work here unchanged.

### 5. Download the results

From a terminal on the Mac:

```bash
scp -i ~/Downloads/<key>.pem "ubuntu@<public IPv4 address>:grpo-vs-gfn-self-correction/bundles/*.tar.gz" ~/Desktop/
```

### 6. Stop paying

- **Stop** the instance (Instance state, then Stop instance) whenever no queue runs. A stopped machine costs only its disk, and it gets a new address when it starts again.
- **Terminate** it once the bundles are on your Mac. That deletes the machine and its disk and ends all charges.

### SageMaker AI instead of EC2

The program's AWS guide uses SageMaker AI notebook instances. They cost about a third more than EC2 but come with JupyterLab and a file browser, and the program's GPU guide walks through the console.

1. **Quota.** In Service Quotas, choose Amazon SageMaker and request 1 on the notebook instance quota for your machine type, for example "ml.g6.xlarge for notebook instance usage". The program's guide shows where.
2. **Notebook instance.** In SageMaker AI (not the older "Amazon SageMaker", as the guide warns), create a notebook instance of type ml.g6.xlarge (L4) or ml.g6e.xlarge (L40S). If neither type appears, ml.g5.xlarge (A10G, 24 GB) also fits. Set the volume size to 50 GB; the default is too small for the model.
3. **Start the queue.** Open JupyterLab, then a terminal:

```bash
cd ~/SageMaker
conda env list
source activate pytorch_p310
export HF_HOME=~/SageMaker/hf-cache
git clone https://github.com/amitkov-sudo/grpo-vs-gfn-self-correction.git
cd grpo-vs-gfn-self-correction
QUEUE_SET=subtb bash scripts/a100_phase1.sh start
```

The second line lists the Python environments. If the PyTorch one has another name, use that name on the third line. `HF_HOME` keeps the downloaded model on the 50 GB volume. After a new login, run the `source activate` line again before `status`.

4. **Download** the bundles from the JupyterLab file browser: right-click a file in `bundles/`, then Download.
5. **Stop** the notebook instance in the SageMaker AI console when the queue has finished, and delete it once the bundles are on your Mac. Closing the browser does not stop the billing.

### When something goes wrong on AWS

- **"You have requested more vCPU capacity than your current vCPU limit":** the quota increase has not come through yet.
- **"Insufficient capacity":** AWS has no free GPU of that type in that zone. Choose another subnet under Network settings, try the other machine type, or try again later.
- **SSH times out after you switch networks:** your address changed. Set the security group's SSH rule to My IP again.
