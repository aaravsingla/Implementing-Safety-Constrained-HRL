Mean ± std over seeds of the last 5 training epochs (SFT reference: 10 deterministic eval episodes per seed).
Cost budget: 25 hazard steps / episode (v1, v2, v3 true-cost); 2 unsafe 50-step segments / episode as judged by the learned cost model (other v3 rows; ≈22 hazard steps).
v3 rows start from the SFT reference, which itself was cloned from 1M steps of unconstrained PPO.

| version   | agent                                               |   seeds | RL steps       | task return   | true cost   | supervisor return   |   near-miss | goals/ep   |
|:----------|:----------------------------------------------------|--------:|:---------------|:--------------|:------------|:--------------------|------------:|:-----------|
| v1        | PPO-Lag + learned RM, cost advantage only centered  |       3 | 1.0M           | 1.2 ± 0.5     | 20.7 ± 4.3  | -1.5 ± 0.6          |        0.16 | 0.8 ± 0.1  |
| v1        | PPO-Lag + hand-coded reward (v1)                    |       3 | 1.0M           | 3.4 ± 2.5     | 28.9 ± 4.4  | 0.4 ± 2.0           |        0.17 | 1.8 ± 1.1  |
| v2        | PPO unconstrained (capable, unsafe)                 |       3 | 1.0M           | 25.2 ± 0.2    | 53.7 ± 1.9  | 19.0 ± 0.1          |        0.34 | 11.9 ± 0.1 |
| v2        | PPO-Lag + hand-coded reward                         |       3 | 1.0M           | 3.8 ± 2.8     | 23.6 ± 5.5  | 0.8 ± 2.3           |        0.19 | 2.0 ± 1.3  |
| v2        | PPO-Lag + learned RM (RLHF from scratch)            |       3 | 1.0M           | 1.7 ± 1.5     | 25.4 ± 7.6  | -1.4 ± 1.0          |        0.18 | 1.1 ± 0.5  |
| v2        | HRL Manager-Worker + PPO-Lag + learned RM           |       3 | 1.0M           | 1.3 ± 0.9     | 28.3 ± 10.0 | -2.1 ± 1.4          |        0.2  | 0.8 ± 0.5  |
| v3        | SFT reference (behavior cloning, eval)              |       3 | 0 (+30k demos) | 26.3 ± 0.3    | 60.4 ± 1.8  | 19.7 ± 0.3          |        0.36 | 12.7 ± 0.4 |
| v3.0      | SFT ref + KL + pessimistic RM + learned cost (full) |       3 | 0.5M           | 25.0 ± 0.2    | 56.7 ± 1.6  | 18.8 ± 0.2          |        0.33 | 11.8 ± 0.1 |
| v3.0      | without KL anchor (β=0)                             |       3 | 0.5M           | 25.2 ± 0.3    | 51.9 ± 5.6  | 19.1 ± 0.6          |        0.33 | 12.0 ± 0.1 |
| v3.0      | without pessimism (κ=0)                             |       3 | 0.5M           | 25.1 ± 0.1    | 51.6 ± 1.8  | 19.0 ± 0.3          |        0.33 | 11.9 ± 0.1 |
| v3.0      | with true cost instead of learned cost              |       3 | 0.5M           | 24.5 ± 1.0    | 49.5 ± 2.2  | 18.9 ± 1.1          |        0.32 | 11.6 ± 0.4 |
| v3.1      | SFT ref + KL + pessimistic RM + learned cost (full) |       3 | 0.8M           | 9.1 ± 2.3     | 43.3 ± 21.2 | 4.2 ± 2.7           |        0.26 | 4.7 ± 1.0  |
| v3.1      | without KL anchor (β=0)                             |       3 | 0.8M           | 6.4 ± 1.9     | 28.6 ± 10.1 | 2.7 ± 1.2           |        0.21 | 3.7 ± 0.7  |
| v3.1      | without pessimism (κ=0)                             |       3 | 0.8M           | 11.8 ± 6.4    | 41.2 ± 12.1 | 6.9 ± 5.4           |        0.28 | 6.2 ± 2.2  |
| v3.1      | with true cost instead of learned cost              |       3 | 0.8M           | 6.8 ± 2.4     | 31.2 ± 3.9  | 3.0 ± 1.8           |        0.21 | 3.8 ± 1.0  |