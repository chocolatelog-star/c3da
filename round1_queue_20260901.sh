#!/usr/bin/env bash
set -u

while pgrep -f '[m]1_element_aware_rgat_treatment_only.py.*chat_round1_' >/dev/null; do
  sleep 30
done

cd /root/CD-C3DA
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p /root/autodl-tmp/tmp

nohup /root/miniconda3/envs/c3da/bin/python3.10 run_reproducible_pipeline.py \
  --recipe /root/CD-C3DA/configs/recipes/rest16_to_laptop14_best_v1.json \
  --run_id batch_round1_16x2_20260901 \
  --output_root /root/autodl-tmp/CD-C3DA-runs \
  --cuda 0 --train_batch_size 16 --eval_batch_size 16 \
  --gradient_accumulation_steps 2 --allow_dirty --skip_validation \
  > /root/autodl-tmp/CD-C3DA-runs/batch_round1_16x2_20260901.log 2>&1 &

nohup /root/miniconda3/envs/c3da/bin/python3.10 run_reproducible_pipeline.py \
  --recipe /root/CD-C3DA/configs/recipes/rest16_to_laptop14_best_v1.json \
  --run_id batch_round1_32x1_20260901 \
  --output_root /root/autodl-tmp/CD-C3DA-runs \
  --cuda 0 --train_batch_size 32 --eval_batch_size 16 \
  --gradient_accumulation_steps 1 --allow_dirty --skip_validation \
  > /root/autodl-tmp/CD-C3DA-runs/batch_round1_32x1_20260901.log 2>&1 &

wait
