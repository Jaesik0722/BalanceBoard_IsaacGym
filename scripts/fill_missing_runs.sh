#!/usr/bin/env bash
# Run only the training-matrix cells that have not completed.
#
# Completion is judged by the "MAX EPOCHS NUM!" marker in logs/<exp>.log, not by
# the run directory existing: a run that crashes part-way leaves a directory
# behind but is not finished, and a run that crashes at import leaves neither.
#
# (run_balance_board_matrix.sh has a broken skip guard -- `[[ -d "path"* ]]`
# does no pathname expansion in bash, so it never matches and the script would
# redo every cell. Fix that there before reusing it; this script is the safe
# way to top up a partially finished matrix.)
#
#   bash scripts/fill_missing_runs.sh            # run what is missing
#   bash scripts/fill_missing_runs.sh --list     # only report what is missing

set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=(1 2 3 4 5)
TASKS=(PID IK Joint)
MODES=(restricted full)
MAX_ITERATIONS=1000

LIST_ONLY=0
[[ "${1:-}" == "--list" ]] && LIST_ONLY=1

mkdir -p logs

missing=()
for task in "${TASKS[@]}"; do
    for mode in "${MODES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            exp="${task}_${mode}_seed${seed}"
            if [[ -f "logs/${exp}.log" ]] && grep -q "MAX EPOCHS NUM!" "logs/${exp}.log"; then
                continue
            fi
            missing+=("${exp}")
        done
    done
done

echo "missing: ${#missing[@]} of 30"
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

echo "gap fill complete: $(date)"
