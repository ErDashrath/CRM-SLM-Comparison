#!/usr/bin/env bash
# Resumes training on an EXISTING, already-provisioned Colab session
# (skips new/upload -- use this after a kernel restart or a failed exec,
# instead of re-running run_colab_training.sh from scratch).
set -euo pipefail
cd "$(dirname "$0")/.."

SESSION="crm-slm-training"
AUTH="--auth=adc"
TIMEOUT=5400

echo "=== Training KD adapter ==="
colab $AUTH exec -s "$SESSION" -f training/colab_train.py --timeout "$TIMEOUT" \
  --env DATASET_PATH=/content/kd_train.jsonl \
  --env ADAPTER_OUT=/content/adapters/kd

echo "=== Downloading KD adapter ==="
mkdir -p adapters
colab $AUTH download -s "$SESSION" /content/adapters/kd.tar.gz adapters/kd.tar.gz
tar -xzf adapters/kd.tar.gz -C adapters --one-top-level=kd
rm adapters/kd.tar.gz

echo "=== Training SFT adapter ==="
colab $AUTH exec -s "$SESSION" -f training/colab_train.py --timeout "$TIMEOUT" \
  --env DATASET_PATH=/content/sft_train.jsonl \
  --env ADAPTER_OUT=/content/adapters/sft

echo "=== Downloading SFT adapter ==="
colab $AUTH download -s "$SESSION" /content/adapters/sft.tar.gz adapters/sft.tar.gz
tar -xzf adapters/sft.tar.gz -C adapters --one-top-level=sft
rm adapters/sft.tar.gz

echo "=== Stopping Colab session ==="
colab $AUTH stop -s "$SESSION"

echo "=== Done. Adapters at adapters/kd/ and adapters/sft/ ==="
