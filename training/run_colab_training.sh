#!/usr/bin/env bash
# Orchestrates the full real-training run on Colab's free T4 (16GB) --
# session lifecycle, uploads, both training runs (kd + sft), adapter
# download, and session cleanup. Run this from the project root:
#
#   bash training/run_colab_training.sh
#
# Requires: ADC auth already completed once
#   (~/Magna/google-cloud-sdk/bin/gcloud auth application-default login
#    --scopes=openid,cloud-platform,userinfo.email,colaboratory)
# -- that step is interactive (opens a browser) and can't be scripted.
#
# Uses --auth=adc explicitly on every colab-cli call: the CLI's own default
# is oauth2, not adc, despite its bundled docs (see project memory /
# SalesIntelligence's Colab notes -- this was a real bug found there).

set -euo pipefail
cd "$(dirname "$0")/.."

SESSION="crm-slm-training"
AUTH="--auth=adc"
TIMEOUT=5400  # 90 min ceiling per training call -- generous, not expected to need all of it on a T4

echo "=== Creating Colab session ($SESSION, T4) ==="
colab $AUTH new --gpu T4 -s "$SESSION"

echo "=== Uploading datasets ==="
colab $AUTH upload -s "$SESSION" data/kd_train.jsonl /content/kd_train.jsonl
colab $AUTH upload -s "$SESSION" data/sft_train.jsonl /content/sft_train.jsonl

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
