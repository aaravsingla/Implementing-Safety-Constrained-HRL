#!/usr/bin/env bash
# Main comparison: 4 flat conditions x 3 seeds, 1M env steps each.
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  echo "train.py --out-dir runs/v2 --reward env --no-constraint --name ppo_env_s$seed --seed $seed"
  echo "train.py --out-dir runs/v2 --reward env --name ppolag_env_s$seed --seed $seed"
  echo "train.py --out-dir runs/v2 --reward oracle --name ppolag_oracle_s$seed --seed $seed"
  echo "train.py --out-dir runs/v2 --reward rm --name ppolag_rm_s$seed --seed $seed"
done | xargs -P "${N_PAR:-10}" -L 1 scripts/_run_one.sh
