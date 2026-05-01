#!/bin/bash
set -e
cd "$(dirname "$0")/.."
LOG="/tmp/overnight_controls.log"
: > "$LOG"
echo "=== START $(date) ===" | tee -a "$LOG"

# Fail-fast: verify venv is usable
if ! .venv/bin/python -c "import torch" 2>/dev/null; then
    echo "FATAL: .venv/bin/python missing or cannot import torch. Abort." | tee -a "$LOG"
    exit 1
fi
echo ".venv OK" | tee -a "$LOG"

# 1. Extra MoE modadd seeds for 20-seed Mann-Whitney (15 new)
echo "=== PHASE 1: extra MoE modadd seeds ===" | tee -a "$LOG"
bash scripts/run_extra_grokking_seeds.sh 2>&1 | tee -a "$LOG"
echo "=== PHASE 1 DONE $(date) ===" | tee -a "$LOG"

# 2. Add-7 param-matched MoE-GLU (h=170)
echo "=== PHASE 2: add-7 param-matched MoE-GLU ===" | tee -a "$LOG"
bash scripts/run_add7_param_matched_moe_glu.sh 2>&1 | tee -a "$LOG"
echo "=== PHASE 2 DONE $(date) ===" | tee -a "$LOG"

# 3. Histogram param-matched GLU and MoE-GLU (h=340)
echo "=== PHASE 3: histogram param-matched GLU + MoE-GLU ===" | tee -a "$LOG"
bash scripts/run_hist_param_matched.sh 2>&1 | tee -a "$LOG"
echo "=== PHASE 3 DONE $(date) ===" | tee -a "$LOG"

echo "=== ALL CONTROLS DONE $(date) ===" | tee -a "$LOG"
