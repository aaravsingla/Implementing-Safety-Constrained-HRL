#!/usr/bin/env bash
# Hierarchical Manager-Worker agent: learned reward (3 seeds) and hand-coded reward (1 seed), 1M steps.
cd "$(dirname "$0")/.."
{
for seed in 0 1 2; do echo "train_hrl.py --out-dir runs/v2 --reward rm --name hrl_ppolag_rm_s$seed --seed $seed"; done
echo "train_hrl.py --out-dir runs/v2 --reward env --name hrl_ppolag_env_s0 --seed 0"
} | xargs -P "${N_PAR:-4}" -L 1 scripts/_run_one.sh
