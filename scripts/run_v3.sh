#!/usr/bin/env bash
# v3: SFT reference -> preference models -> PPO-Lagrangian + KL fine-tuning, 3 seeds.
#   full      : KL anchor (beta=0.1) + pessimistic ensembles (kappa=1) + learned cost (Safe-RLHF)
#   no_kl     : beta = 0                       -> isolates the KL anchor
#   no_pess   : kappa = 0                      -> isolates pessimism
#   true_cost : constraint on the true hazard cost instead of the learned cost model
# VERSION=v3.0 reproduces the first attempt (default optimizer settings, 25 epochs);
# VERSION=v3.1 (default) adds the plasticity fixes found by diagnosing v3.0 (see JOURNEY.md).
cd "$(dirname "$0")/.."
VERSION=${VERSION:-v3.1}
mkdir -p logs/v3/$VERSION
export LOG_DIR=logs/v3/$VERSION
for seed in 0 1 2; do
  [ -f runs/v3/reference_s$seed/policy.pt ] || .venv/bin/python -W ignore make_reference.py --seed $seed > logs/v3/reference_s$seed.log 2>&1 &
done; wait
if [ "$VERSION" = "v3.0" ]; then
  COMMON="--out-dir runs/v3/v3.0"; LR_RM=0.01; LR_ENV=0.01
else
  COMMON="--out-dir runs/v3/v3.1 --epochs 40 --pi-lr 1e-3 --update-iters 20 --init-log-std -0.5"; LR_RM=0.5; LR_ENV=0.05
fi
for seed in 0 1 2; do
  echo "train_safe_rlhf.py $COMMON --lambda-lr $LR_RM --name v3_full_s$seed --seed $seed"
  echo "train_safe_rlhf.py $COMMON --lambda-lr $LR_RM --kl-coef 0 --name v3_no_kl_s$seed --seed $seed"
  echo "train_safe_rlhf.py $COMMON --lambda-lr $LR_RM --pessimism 0 --name v3_no_pess_s$seed --seed $seed"
  echo "train_safe_rlhf.py $COMMON --lambda-lr $LR_ENV --cost-source env --name v3_true_cost_s$seed --seed $seed"
done | xargs -P "${N_PAR:-12}" -L 1 scripts/_run_one.sh
