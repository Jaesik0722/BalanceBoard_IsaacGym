#!/usr/bin/env bash
# Training matrix at the extended budget: 3 tasks x 2 observation modes x 3 seeds
# at 3000 PPO iterations.
#
# The 1000-iteration runs are kept, not replaced: the pair of budgets is itself
# the result. A pilot showed joint-space going from 0% to 52% deterministic
# completion between the two budgets while IK was already saturated at 1000, so
# a single-budget comparison measures convergence speed rather than ceiling.
#
# Three seeds rather than five is a compute-budget decision, taken before the
# runs and stated in the paper; the effect size seen in the pilot is large
# enough to resolve at n=3.
#
# Completion is judged by the marker in logs/<exp>.log, so re-running tops up a
# partial matrix instead of redoing it.
#
#   bash scripts/run_matrix_3k.sh          # run what is missing
#   bash scripts/run_matrix_3k.sh --list   # report what is missing

set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=(1 2 3)
TASKS=(PID IK Joint)
MODES=(restricted full)
MAX_ITERATIONS=3000

LIST_ONLY=0
[[ "${1:-}" == "--list" ]] && LIST_ONLY=1

mkdir -p logs

missing=()
for task in "${TASKS[@]}"; do
    for mode in "${MODES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            exp="${task}_${mode}_3k_seed${seed}"
            if [[ -f "logs/${exp}.log" ]] && grep -q "MAX EPOCHS NUM!" "logs/${exp}.log"; then
                continue
            fi
            missing+=("${exp}")
        done
    done
done

echo "missing: ${#missing[@]} of 18"
for exp in "${missing[@]}"; do echo "  ${exp}"; done
[[ ${LIST_ONLY} -eq 1 ]] && exit 0
[[ ${#missing[@]} -eq 0 ]] && { echo "nothing to do"; exit 0; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate isaacgym

for exp in "${missing[@]}"; do
    task="${exp%%_*}"
    rest="${exp#*_}"
    mode="${rest%%_*}"
    seed="${exp##*seed}"
    if [[ "${mode}" == "full" ]]; then full_state="true"; else full_state="false"; fi

    echo "=== ${exp} : starting $(date) ==="
    python train.py \
        task="BalanceBoard${task}" \
        task.env.fullStateObservation="${full_state}" \
        seed="${seed}" \
        headless=True \
        max_iterations="${MAX_ITERATIONS}" \
        experiment="${exp}" \
        2>&1 | tee "logs/${exp}.log"
    echo "=== ${exp} : finished $(date) ==="
done

echo "3k matrix complete: $(date)"
