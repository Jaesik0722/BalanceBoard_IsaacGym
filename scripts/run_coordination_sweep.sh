#!/bin/bash
# Coordination sweep. The force ladder was fixed by measuring where the policies
# still survive a push: 1-3 N is comfortably survivable, 5 N is marginal (about half
# the trials fall within 2 s) and 10 N is uniformly fatal. 0 N is the unperturbed
# control that the pushed conditions are read against.
cd "$(dirname "$0")/.."
source "${CONDA_PREFIX:-$HOME/anaconda3}/etc/profile.d/conda.sh" 2>/dev/null \
  || source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null
conda activate isaacgym 2>/dev/null
OUT=results/coordination
mkdir -p "$OUT"
echo "=== coordination sweep start $(date '+%m-%d %H:%M') ==="
for task in PID IK Joint; do
  for mode in restricted full; do
    for axis in pitch roll; do
      for f in 0 1 3 5; do
        out="$OUT/${task}_${mode}_${axis}_${f}N.csv"
        [ -s "$out" ] && { echo "skip $out"; continue; }
        timeout 900 python scripts/coordination_eval.py --task "$task" --obs-mode "$mode" \
          --axis "$axis" --force "$f" --trials 10 --out "$out" 2>&1 | grep -E "stabilised|Error|error"
      done
    done
  done
done
echo "=== coordination sweep done $(date '+%m-%d %H:%M') ==="
