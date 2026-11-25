#!/usr/bin/env bash
# Preliminary Phase 1 for one free Kaggle session: two T4 GPUs, 12 hours at most.
#
#   PAIR=main bash scripts/kaggle_prelim.sh           T5 against T4, the main comparison (first session)
#   PAIR=own  bash scripts/kaggle_prelim.sh           T3 against T2, the model's own attempts only (second session)
#   PAIR=main bash scripts/kaggle_prelim.sh summary   the result tables again
#
# The same question, conditions, model and measures as Phase 1 (docs/a100-phase1-runbook.md), at a smaller scale:
# 8 starts per step instead of 16, one seed instead of two, 4 sampled answers per evaluation start instead of 8, and
# 20 location problems instead of 50. With two GPUs the two runs go side by side, one per GPU; with one GPU they run
# in turn. Each run gets the session time left as --max-hours, so it stops early, evaluates and saves before Kaggle
# ends the session. docs/kaggle-prelim.md explains the choices.
#
# Settings through environment variables, with their defaults:
#   PAIR=main  SEED=0  MODEL=Qwen/Qwen3-1.7B  PROMPTS=8  STEPS=60  EVAL_EVERY=20  EVAL_SAMPLES=4  LOCATION_N=20
#   GEN_BATCH=64  MICRO_BATCH=2  SESSION_HOURS=11.5  PYTHON=python3
#   SESSION_HOURS  hours this script may use. Kaggle stops a session at 12 hours, and setup counts against them
#   TOKENIZER=     only for an offline check whose cached tokenizer lacks a chat template
#   SMOKE=1        both runs with the tiny random model on the CPU, two steps each: a code check
set -euo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."
T0=$(date +%s)

PY=${PYTHON:-$(command -v python3 || command -v python)}
PAIR=${PAIR:-main}
SEED=${SEED:-0}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
PROMPTS=${PROMPTS:-8}
STEPS=${STEPS:-60}
EVAL_EVERY=${EVAL_EVERY:-20}
EVAL_SAMPLES=${EVAL_SAMPLES:-4}
LOCATION_N=${LOCATION_N:-20}
GEN_BATCH=${GEN_BATCH:-64}
MICRO_BATCH=${MICRO_BATCH:-2}
SESSION_HOURS=${SESSION_HOURS:-11.5}
SMOKE=${SMOKE:-}
TOKENIZER=${TOKENIZER:-}

case "$PAIR" in
    main) RUNS=("T5:--method tb --nudge 0.5" "T4:--method grpo --nudge 0.5") ;;
    own)  RUNS=("T3:--method grpo --anchor" "T2:--method grpo") ;;
    *)    echo "PAIR must be main or own"; exit 1 ;;
esac
if [ -n "$SMOKE" ]; then
    PREFIX="smoke-prelim-"
    COMMON=(--task fix --smoke --eval-samples 2 --location-n 2)
    STEPS=2; EVAL_EVERY=2
else
    PREFIX="prelim-"
    COMMON=(--task fix --model "$MODEL" --prompts "$PROMPTS" --steps "$STEPS" --eval-every "$EVAL_EVERY"
            --eval-samples "$EVAL_SAMPLES" --location-n "$LOCATION_N" --gen-batch "$GEN_BATCH" --micro-batch "$MICRO_BATCH")
fi
if [ -n "$TOKENIZER" ]; then COMMON+=(--tokenizer "$TOKENIZER"); fi
mkdir -p outputs bundles

names() {
    for entry in "${RUNS[@]}"; do echo "${PREFIX}${entry%%:*}-s${SEED}"; done
}

summary() {
    # shellcheck disable=SC2046
    "$PY" scripts/summarize_runs.py --prefix "$PREFIX" --steps "$STEPS" --eval-every "$EVAL_EVERY" --expect $(names)
}

hours_left() {
    # session hours not yet used, passed to train.py as --max-hours; at least 0.01 so the option stays valid
    awk -v s="$SESSION_HOURS" -v t0="$T0" -v now="$(date +%s)" -v part="${1:-1}" \
        'BEGIN { h = (s - (now - t0) / 3600) * part; if (h < 0.01) h = 0.01; printf "%.3f", h }'
}

finished() {
    compgen -G "outputs/$1/adapter" > /dev/null || compgen -G "outputs/$1-*-*/adapter" > /dev/null
}

setup() {
    if [ -n "$SMOKE" ]; then return; fi
    "$PY" -m pip install -q -r requirements.txt
    # Kaggle ships torchao 0.10, and recent PEFT refuses to add LoRA layers next to a torchao older than 0.16.
    # This repository does not use torchao.
    "$PY" -m pip uninstall -q -y torchao > /dev/null 2>&1 || true
}

prefetch() {
    # Download the data and the model once, so the two runs do not fetch the same files at the same time.
    "$PY" -c "import sys; sys.path.insert(0, '.'); from selfcorrect import data; \
data.load_gsm8k('train'); data.load_gsm8k('test'); data.sc_split()"
    if [ -z "$SMOKE" ]; then
        "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" > /dev/null
    fi
}

run_one() {
    # $1 condition, $2 its train.py flags, $3 GPU index (empty for the CPU), $4 hours for --max-hours
    local cond=$1 gpu=$3 hours=$4 name
    local -a flags
    read -r -a flags <<< "$2"
    name="${PREFIX}${cond}-s${SEED}"
    if finished "$name"; then echo "$(date '+%F %T') skip $name: finished before"; return 0; fi
    echo "$(date '+%F %T') start $name on GPU ${gpu:-none}, --max-hours $hours; log: outputs/$name.out"
    if CUDA_VISIBLE_DEVICES=$gpu "$PY" train/train.py "${COMMON[@]}" "${flags[@]}" --seed "$SEED" --run "$name" \
            --max-hours "$hours" > "outputs/$name.out" 2>&1; then
        echo "$(date '+%F %T') finished $name"
    else
        echo "$(date '+%F %T') FAILED $name; see outputs/$name.out"
        return 1
    fi
}

pack() {
    local stamp files
    stamp=$(date +%Y%m%d-%H%M%S)
    summary > "outputs/${PREFIX}summary.md"
    files=$(find outputs -maxdepth 2 -path "outputs/${PREFIX}*" \
        \( -name config.json -o -name log.jsonl -o -name 'eval_step*.jsonl' -o -name '*.out' \) | sort)
    # shellcheck disable=SC2086
    tar -czf "bundles/${PREFIX}${PAIR}-${stamp}.tar.gz" "outputs/${PREFIX}summary.md" $files
    echo "bundle: bundles/${PREFIX}${PAIR}-${stamp}.tar.gz ($(du -h "bundles/${PREFIX}${PAIR}-${stamp}.tar.gz" | cut -f1))"
}

if [ "${1:-}" = "summary" ]; then summary; exit 0; fi

if [ -n "$SMOKE" ]; then
    GPUS=0
else
    setup
    GPUS=$("$PY" -c "import torch; print(torch.cuda.device_count())")
    if [ "$GPUS" -lt 1 ]; then echo "No GPU found. In Kaggle's settings, choose Accelerator: GPU T4 x2."; exit 1; fi
    "$PY" -c "import torch; print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
fi
echo "CPU checks: offline tests and the smoke run"
"$PY" -m pytest -q tests
"$PY" train/train.py --smoke ${TOKENIZER:+--tokenizer "$TOKENIZER"} --out outputs/checks > outputs/checks.log 2>&1 \
    || { echo "The smoke run failed; see outputs/checks.log"; exit 1; }
prefetch

status=0
first=${RUNS[0]}
second=${RUNS[1]}
if [ "$GPUS" -ge 2 ]; then
    hours=$(hours_left)
    run_one "${first%%:*}" "${first#*:}" 0 "$hours" &
    p0=$!
    run_one "${second%%:*}" "${second#*:}" 1 "$hours" &
    p1=$!
    wait "$p0" || status=1
    wait "$p1" || status=1
else
    gpu=0
    if [ -n "$SMOKE" ]; then gpu=""; fi
    run_one "${first%%:*}" "${first#*:}" "$gpu" "$(hours_left 0.5)" || status=1
    run_one "${second%%:*}" "${second#*:}" "$gpu" "$(hours_left)" || status=1
fi
pack
echo
cat "outputs/${PREFIX}summary.md"
exit "$status"
