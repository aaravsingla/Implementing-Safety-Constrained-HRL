#!/usr/bin/env bash
# Ablations at 600k steps (30 epochs), 1 seed each: query selection, label budget,
# label noise, reward-model input features, PID Lagrangian.
cd "$(dirname "$0")/.."
E="--epochs 30 --seed 0"
{
echo "train.py --out-dir runs/v2 --reward rm $E --name abl_activeq_s0"
echo "train.py --out-dir runs/v2 --reward rm $E --random-queries --name abl_randomq_s0"
echo "train.py --out-dir runs/v2 --reward rm $E --label-budget 500 --name abl_budget500_s0"
echo "train.py --out-dir runs/v2 --reward rm $E --label-noise 0.3 --name abl_noise0.3_s0"
echo "train.py --out-dir runs/v2 --reward rm $E --rm-features summary --rm-hidden 64 --rm-weight-decay 1e-2 --rm-steps-first 500 --rm-steps 150 --name abl_rmsummary_s0"
echo "train.py --out-dir runs/v2 --reward env $E --lagrange-mode pid --name abl_pid_env_s0"
} | xargs -P "${N_PAR:-6}" -L 1 scripts/_run_one.sh
