#!/usr/bin/env bash
set -o pipefail

ROOT_DIR="/home/hello/research_project/event+SNN+TTC"
PY="$ROOT_DIR/EV-TTC-main/.venv/bin/python"
OUT_DIR="$ROOT_DIR/EV-TTC-SNN-main/debug_sets/skatepark_multi_n_ttc"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$ROOT_DIR" || exit 1

status_file="$OUT_DIR/run_status.md"
{
  echo "# Skatepark multi-N TTC run status"
  echo
  echo "- start: $(date '+%F %T')"
  echo "- out_dir: $OUT_DIR"
} > "$status_file"

for N in 5000 10000 15000 20000; do
  echo "[N=$N] build start $(date '+%F %T')" | tee -a "$LOG_DIR/N${N}.log"
  if "$PY" EV-TTC-SNN-main/snn_ttc/tools/build_skatepark_multi_n_ttc.py \
      --sequence spot_outdoor_day_skatepark_1 \
      --event-counts "$N" \
      --roi-size 128 \
      --steps-per-roi 10 \
      --max-step-duration-ms 10 \
      --audit-fixed-window-ms 10 \
      --seed 42 \
      --resume \
      --continue-on-error \
      --out-dir "$OUT_DIR" 2>&1 | tee -a "$LOG_DIR/N${N}.log"; then
    echo "- N=$N: build ok" >> "$status_file"
  else
    echo "- N=$N: build failed, continuing" >> "$status_file"
  fi
done

if "$PY" EV-TTC-SNN-main/snn_ttc/tools/audit_skatepark_multi_n_ttc.py --out-dir "$OUT_DIR" 2>&1 | tee "$LOG_DIR/audit.log"; then
  echo "- audit: ok" >> "$status_file"
else
  echo "- audit: failed" >> "$status_file"
fi

if "$PY" EV-TTC-SNN-main/snn_ttc/tools/visualize_skatepark_multi_n_ttc.py --out-dir "$OUT_DIR" 2>&1 | tee "$LOG_DIR/visualize.log"; then
  echo "- visualize: ok" >> "$status_file"
else
  echo "- visualize: failed" >> "$status_file"
fi

echo "- finish: $(date '+%F %T')" >> "$status_file"
