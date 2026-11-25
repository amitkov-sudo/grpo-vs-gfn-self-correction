#!/usr/bin/env bash
# Queues for the program's A100 (40 GB): the fix task on Qwen3-1.7B.
#
#   bash scripts/a100_phase1.sh start            install, run the CPU checks, then start the queue in the background
#   bash scripts/a100_phase1.sh status           progress, latest evaluations and time left
#   bash scripts/a100_phase1.sh bundle           pack logs and evaluations into bundles/ for download
#   bash scripts/a100_phase1.sh bundle adapters  the same, plus the LoRA adapters (about 140 MB per run)
#   bash scripts/a100_phase1.sh stop             stop the queue and the run in progress
#
# Three queues, chosen with QUEUE_SET:
#   phase1 (default)  T5 and T4 first, the main comparison, then T3 and T2. Every evaluation adds 8 sampled answers
#                     per start (pass@8, majority vote); step 0 and the last step add the error at step 2, 3 and 4.
#   poison            the poisoning test: GRPO (T2) and trajectory balance on own attempts, with the reward flipped
#                     on 25% and then 10% of wrong starts. Unpoisoned references are T2 and T3 in phase1.
#   attack            the attacker-replies test: trajectory balance and GRPO, with 4 of the 8 answers on 25% and then
#                     5% of wrong starts written after " That step is correct." and rewarded for a wrong answer.
# Each queue runs seed 0 and then seed 1. A finished run (outputs/<run>/adapter exists) is skipped, so `start` after
# a stop picks up where the queue left off; an unfinished run starts again from step 0. After every run the script
# packs a new bundle.
#
# Settings through environment variables, with their defaults:
#   QUEUE_SET=phase1  SEEDS="0 1"  MODEL=Qwen/Qwen3-1.7B  STEPS=60  EVAL_EVERY=20  EVAL_SAMPLES=8  LOCATION_N=50
#   GEN_BATCH=64  MICRO_BATCH=4  PYTHON=python3
#   TOKENIZER=   only for an offline check whose cached tokenizer lacks a chat template
#   SMOKE=1      the whole queue with the tiny random model on the CPU, two steps per run: a code check
set -euo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

PY=${PYTHON:-$(command -v python3 || command -v python)}
SEEDS=${SEEDS:-"0 1"}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
STEPS=${STEPS:-60}
EVAL_EVERY=${EVAL_EVERY:-20}
EVAL_SAMPLES=${EVAL_SAMPLES:-8}
LOCATION_N=${LOCATION_N:-50}
GEN_BATCH=${GEN_BATCH:-64}
MICRO_BATCH=${MICRO_BATCH:-4}
QUEUE_SET=${QUEUE_SET:-phase1}
SMOKE=${SMOKE:-}
TOKENIZER=${TOKENIZER:-}

case "$QUEUE_SET" in
    phase1)
        QUEUE=("T5:--method tb --nudge 0.5" "T4:--method grpo --nudge 0.5" "T3:--method grpo --anchor" "T2:--method grpo")
        EXTRA=(--location-n "$LOCATION_N") ;;
    poison)
        QUEUE=("T2-p25:--method grpo --poison 0.25" "TB-p25:--method tb --poison 0.25"
               "T2-p10:--method grpo --poison 0.1" "TB-p10:--method tb --poison 0.1")
        EXTRA=() ;;
    attack)
        QUEUE=("TB-a25:--method tb --attack 0.25" "T2-a25:--method grpo --attack 0.25"
               "TB-a5:--method tb --attack 0.05" "T2-a5:--method grpo --attack 0.05")
        EXTRA=() ;;
    *)
        echo "QUEUE_SET must be phase1, poison or attack"; exit 1 ;;
esac
if [ -n "$SMOKE" ]; then
    PREFIX="smoke-${QUEUE_SET}-"
    COMMON=(--task fix --smoke --eval-samples 2)
    if [ "$QUEUE_SET" = phase1 ]; then COMMON+=(--location-n 2); fi
    STEPS=2; EVAL_EVERY=2
else
    PREFIX="${QUEUE_SET}-"
    COMMON=(--task fix --model "$MODEL" --steps "$STEPS" --eval-every "$EVAL_EVERY" --eval-samples "$EVAL_SAMPLES"
            --gen-batch "$GEN_BATCH" --micro-batch "$MICRO_BATCH" "${EXTRA[@]+"${EXTRA[@]}"}")
fi
if [ -n "$TOKENIZER" ]; then COMMON+=(--tokenizer "$TOKENIZER"); fi

mkdir -p outputs bundles
LOG="outputs/${PREFIX}queue.log"
PIDFILE="outputs/${PREFIX}queue.pid"

names() {
    for seed in $SEEDS; do
        for entry in "${QUEUE[@]}"; do echo "${PREFIX}${entry%%:*}-s${seed}"; done
    done
}

summary() {
    # shellcheck disable=SC2046
    "$PY" scripts/summarize_runs.py --prefix "$PREFIX" --steps "$STEPS" --eval-every "$EVAL_EVERY" --expect $(names)
}

finished() {
    compgen -G "outputs/$1/adapter" > /dev/null || compgen -G "outputs/$1-*-*/adapter" > /dev/null
}

setup() {
    if [ -n "$SMOKE" ]; then return; fi
    if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "PyTorch with CUDA not found; installing it"
        "$PY" -m pip install -q torch || "$PY" -m pip install -q --user torch
    fi
    "$PY" -m pip install -q -r requirements.txt || "$PY" -m pip install -q --user -r requirements.txt
    # PEFT refuses to add LoRA layers next to a torchao older than 0.16; this repository does not use torchao.
    if "$PY" - <<'EOF'
import importlib.metadata as m, sys
try:
    v = tuple(int(x) for x in m.version("torchao").split(".")[:2])
except Exception:
    sys.exit(1)
sys.exit(0 if v < (0, 16) else 1)
EOF
    then "$PY" -m pip uninstall -q -y torchao; fi
    "$PY" -c "import torch; print('GPU:', torch.cuda.get_device_name(0), '| bfloat16:', torch.cuda.is_bf16_supported())"
}

pack() {
    local stamp files
    stamp=$(date +%Y%m%d-%H%M%S)
    summary > "outputs/${PREFIX}summary.md"
    files=$(find outputs -maxdepth 2 -path "outputs/${PREFIX}*" \
        \( -name config.json -o -name log.jsonl -o -name 'eval_step*.jsonl' \) | sort)
    if [ "${1:-}" = "adapters" ]; then
        files="$files $(find outputs -maxdepth 2 -type d -path "outputs/${PREFIX}*" -name adapter | sort)"
    fi
    if [ -f "$LOG" ]; then files="$LOG $files"; fi
    # shellcheck disable=SC2086
    tar -czf "bundles/${PREFIX}${stamp}.tar.gz" "outputs/${PREFIX}summary.md" $files
    echo "bundle: bundles/${PREFIX}${stamp}.tar.gz ($(du -h "bundles/${PREFIX}${stamp}.tar.gz" | cut -f1))"
    # keep the three newest bundles without adapters
    ls -1t bundles/"${PREFIX}"*.tar.gz 2>/dev/null | tail -n +4 | while read -r old; do
        tar -tzf "$old" | grep -q '/adapter/' || rm -f "$old"
    done
}

run_queue() {
    echo "$(date '+%F %T') queue: $(names | tr '\n' ' ')"
    for seed in $SEEDS; do
        for entry in "${QUEUE[@]}"; do
            cond=${entry%%:*}
            read -r -a flags <<< "${entry#*:}"
            name="${PREFIX}${cond}-s${seed}"
            if finished "$name"; then echo "$(date '+%F %T') skip $name: finished before"; continue; fi
            echo "$(date '+%F %T') start $name"
            if "$PY" train/train.py "${COMMON[@]}" "${flags[@]}" --seed "$seed" --run "$name"; then
                echo "$(date '+%F %T') finished $name"
            else
                echo "$(date '+%F %T') FAILED $name; the queue moves on"
            fi
            pack || echo "$(date '+%F %T') packing a bundle failed; the queue moves on"
        done
    done
    echo "$(date '+%F %T') queue finished. Download the newest file in bundles/ before the grant ends."
}

running() {
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

other_queue() {
    # one GPU, one queue at a time: prints the pid file of another running queue, if any
    for f in outputs/*queue.pid; do
        [ -f "$f" ] && [ "$f" != "$PIDFILE" ] && kill -0 "$(cat "$f")" 2>/dev/null && echo "$f"
    done
    return 0
}

case "${1:-}" in
    start)
        if running; then echo "The queue is already running (pid $(cat "$PIDFILE")). See: bash $SCRIPT status"; exit 0; fi
        busy=$(other_queue)
        if [ -n "$busy" ]; then echo "Another queue is running ($busy). Wait for it or stop it first."; exit 1; fi
        setup
        echo "CPU checks: offline tests and the smoke run"
        "$PY" -m pytest -q tests
        "$PY" train/train.py --smoke ${TOKENIZER:+--tokenizer "$TOKENIZER"} --out outputs/checks > outputs/checks.log 2>&1 \
            || { echo "The smoke run failed; see outputs/checks.log"; exit 1; }
        nohup bash "$SCRIPT" run-queue >> "$LOG" 2>&1 &
        echo $! > "$PIDFILE"
        echo "Queue started (pid $!). Log: $LOG"
        echo "Check progress with: bash $SCRIPT status"
        ;;
    run-queue)
        run_queue
        ;;
    status)
        if running; then echo "Queue running (pid $(cat "$PIDFILE"))."; else echo "Queue not running."; fi
        if [ -f "$LOG" ]; then echo "--- last lines of $LOG"; tail -n 4 "$LOG"; fi
        echo
        summary
        if command -v nvidia-smi > /dev/null; then
            echo; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
        fi
        ;;
    bundle)
        pack "${2:-}"
        ;;
    stop)
        if running; then
            pid=$(cat "$PIDFILE")
            pkill -TERM -P "$pid" 2>/dev/null || true
            kill "$pid" 2>/dev/null || true
            echo "Stopped the queue (pid $pid)."
        else
            echo "Queue not running."
        fi
        rm -f "$PIDFILE"
        ;;
    *)
        awk 'NR > 1 && /^set -euo pipefail/ {exit} NR > 1 {print}' "$SCRIPT"
        exit 1
        ;;
esac
