#!/usr/bin/env bash
# Full training matrix for the BalanceBoard Applied Sciences resubmission.
#
# 3 tasks (PID, IK, Joint) x 2 observation modes (restricted, full) x 5 seeds
# = 30 sequential runs. Benchmarked on this machine (RTX 2070) at ~33-44k fps
# with 4096 envs / horizon 32 -> roughly 50-70 min per 1000-iteration run,
# so budget ~25-35 hours of continuous GPU time for the whole matrix.
#
# Decisions A/B/C from HANDOFF.md are already baked into the code/config
# (right-foot termination, per-env reset offsets, Joint rewardWeight=-5,
# single max_epochs shared across obs modes). This script only fixes
# Decision D: 5 seeds per condition, listed explicitly below (distinct from
# the old seed=42 pre-fix runs archived in archive/pre-fix-runs/, so results
# are never confused).
#
# Usage:
#   bash scripts/run_balance_board_matrix.sh                 # run everything
#   bash scripts/run_balance_board_matrix.sh PID              # just one task
#   bash scripts/run_balance_board_matrix.sh PID restricted    # task + mode
#
# Each run's full resolved config (incl. git-independent snapshot of every
# hyperparameter), seed, and iteration/transition count is auto-saved by
# rl_games to runs/<experiment>/config.yaml -- no separate manifest needed.
# Console output is additionally tee'd to logs/<experiment>.log.

set -uo pipefail  # no -e: one failed run should not kill the whole matrix

cd "$(dirname "$0")/.."  # isaacgymenvs/

SEEDS=(1 2 3 4 5)
TASKS=(PID IK Joint)
MODES=(restricted full)
MAX_ITERATIONS=1000

TASK_FILTER="${1:-}"
MODE_FILTER="${2:-}"

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate isaacgym

run_one () {
    local task="$1" mode="$2" seed="$3"
    local full_state
    if [[ "$mode" == "full" ]]; then full_state="true"; else full_state="false"; fi
    local exp="${task}_${mode}_seed${seed}"

    if [[ -d "runs/${exp}"* ]]; then
        echo "SKIP (already exists): ${exp}"
        return
    fi

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
}

for task in "${TASKS[@]}"; do
    [[ -n "$TASK_FILTER" && "$task" != "$TASK_FILTER" ]] && continue
    for mode in "${MODES[@]}"; do
        [[ -n "$MODE_FILTER" && "$mode" != "$MODE_FILTER" ]] && continue
        for seed in "${SEEDS[@]}"; do
            run_one "$task" "$mode" "$seed"
        done
    done
done

echo "Matrix complete: $(date)"
