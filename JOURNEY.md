# Project Journey: from a broken prototype to Safe-RLHF on a control task

This is a working log of how the project developed, with the obstacles hit along the way. Each version has its own code,
runs and results, so every claim below can be traced to an artifact:

| Version | What it is | Code | Runs / logs | Results |
|---|---|---|---|---|
| **v0** | Original prototype (Jan 2026) | `history/v0_original/` | n/a | `history/v0_original/*.png` |
| **v1** | Rebuild: PPO-Lagrangian + preference reward model + HRL on Safety-Gymnasium | current code with `--cost-adv-norm center` | `runs/v1/`, `logs/v1/` | `results/v2/cost_adv_fix.png`, `results/journey_table.md` |
| **v2** | v1 + standardized cost advantage, ablations, reward-model diagnostics | `train.py`, `train_hrl.py` | `runs/v2/`, `logs/v2/` | `results/v2/` |
| **v3.0** | LLM-style RLHF: SFT reference → reward and cost models → PPO + KL; pessimistic ensembles; learned cost (Safe-RLHF) | `make_reference.py`, `train_safe_rlhf.py`, `srl/safe_rlhf.py` | `runs/v3/v3.0/`, `logs/v3/v3.0/` | `results/v3/v3.0/` |
| **v3.1** | v3.0 + plasticity fixes (policy lr, PPO iterations, exploration reset, λ step size), 800k steps | same, `VERSION=v3.1 scripts/run_v3.sh` | `runs/v3/v3.1/`, `logs/v3/v3.1/` | `results/v3/v3.1/` |

Cross-version numbers: [`results/journey_table.md`](results/journey_table.md).

---

## v0: the original prototype (and why it failed review)

The project was on my CV as *"Safety-Constrained Hierarchical RL with Human Feedback"*. A careful code review
(`history/v0_original/`, about 200 lines) found that **none of the three headline claims held up in the code**:

| Claim | What the code actually did |
|---|---|
| RLHF reward model trained on preference pairs | The reward network was **never trained**: no preference data, no Bradley-Terry loss, no optimizer. The policy maximized a *randomly initialized* network, and the environment reward was discarded (`next_obs, _, ...`). |
| Manager-Worker HRL with temporal sub-goals | The manager emitted **one** sub-goal per episode. The sub-goal had no semantics, and the worker had no reward for reaching it. |
| CMDP + Lagrangian | λ moved in the right direction, but the penalty was the same for every step of an episode, so there was no credit assignment to the steps that actually violated. |
| "PPO-style update" | Single-episode REINFORCE with no critic. |
| "Learns to reach the target" | The repo's own figure shows the agent hugging the walls and never reaching the goal. The logged reward was **+169**, impossible for a `−distance` reward, which proves it was the random network's output. |

**Lesson:** I had written a description of what I wanted the project to be, not what it was. An interviewer who opens the
link would find this in minutes. The rebuild started from the rule *every claim must map to code and a measured number*.

---

## v1: rebuilding it properly (2026-10-09)

### Design decisions
* **Benchmark instead of a toy environment:** Safety-Gymnasium `SafetyPointGoal1` (60-D lidar observations, 1000-step
  episodes, randomized layouts). It's a standard safe-RL benchmark, so the numbers are comparable to published work.
* **Written from scratch, not OmniSafe:** a CleanRL-style PPO-Lagrangian (~300 lines) with separate reward and cost critics
  and GAE for each. Two reasons: OmniSafe's pinned dependencies break on Apple Silicon, and I need to be able to defend every line.
* **A "nuanced human preference" the hand-coded reward lacks:** the hidden supervisor also penalizes *grazing* hazards
  (being within 0.3 m), not just entering them. The agent never sees this signal. It's only reachable through preference queries.
* **RLHF pipeline (Christiano et al. 2017):**
  - 50-step segments, compared in pairs
  - a simulated annotator that answers wrong 10% of the time
  - a Bradley-Terry ensemble of 3 networks
  - active queries chosen by ensemble disagreement
  - a 2000-label budget
* **HRL:** the manager emits an egocentric sub-goal every 10 steps and the worker gets an intrinsic progress reward.
  Both levels are Lagrangian-constrained, and both train on-policy to avoid HIRO's off-policy relabeling problem.

### Obstacles
1. **Dependency problems:**
   - `safety-gymnasium` pins `pygame==2.1.0`, which has no wheel for Python ≥3.11, and `mujoco==2.3.x`.
   - Python 3.11 then failed with a dataclass "mutable default" error inside safety-gymnasium.
   - Fix: Python 3.10 via `uv`, plus a dependency override for pygame.
2. **Compute on a fanless laptop:**
   - 22 parallel runs on a 16 GB MacBook Air ran into heavy memory compression, with 37% of CPU time going to the system.
   - Throughput was about 4× below what I expected.
   - Fix: lower parallelism and queue runs in batches.
3. **Reward model overfitting, found mid-run:**
   - Training loss was 0.008 even though 10% of labels were noise, so the floor should be about 0.3.
   - Held-out accuracy fell from 0.82 to 0.69.
   - I paused and ran offline experiments on 30k exploration steps. Changing regularization or network size didn't move held-out accuracy (it stayed ~0.66).
4. **Reward identifiability:**
   - Even a regressor trained *directly* on 20k true rewards reached only 0.42 correlation from raw lidar, 0.59 with lidar-summary features.
   - This is a property of the observation space, not of preference learning. Good to know before blaming the method.
5. **Tooling:** macOS `xargs -I` has a 255-byte replacement limit, which silently dropped 4 ablation jobs.
   Fix: a small `_run_one.sh` wrapper with `xargs -L 1`.

### v1 result
The constraint worked: cost fell from 54 (unconstrained PPO) to about the budget of 25. But the **constrained agents
barely learned the task** (return about 1–5 vs 25 for PPO).

---

## v2: a hypothesis, a "fix", and an honest null result

**Hypothesis:** in `A = (Â_R − λÂ_C)/(1+λ)`, Â_R was standardized but Â_C was only centered. Discounted cost
advantages are O(10), so with λ≈3 the cost term was about 30× the reward term and should swamp the policy gradient.
OmniSafe standardizes both.

**Result:** standardizing Â_C improved *early* learning in some seeds, but **final metrics were within seed noise**
(task return 3.8 ± 2.8 vs 3.4 ± 2.5). λ simply settled at a higher value to compensate.
**Lesson:** a mechanism that is clearly "wrong" on paper doesn't have to matter in practice. Test fixes across seeds before claiming them.

### v2 findings
* **The constraint works:** cost is held at the budget (18–25 vs 54) and near-misses drop by 45%. The price is
  task return (25 → ~4). This is the reward-safety trade-off, made worse by the short 1M-step budget.
* **RLHF from scratch recovers about 45%** of the hand-coded agent's task return with 2000 labels.
* **Reward-model overoptimization (the most interesting result):**
  - The lidar-summary reward model had the *best* held-out accuracy (0.79) and correlated +0.60 with the true reward on fixed data.
  - On the policy's own rollouts, its correlation went from **+0.53 to −0.83**.
  - Held-out accuracy can't detect this, because the labeled pairs come from older policies.
* **HRL gave no gain** at this scale, consistent with the GUARD benchmark.
* **PID-Lagrangian:** I first wrote that it "damps oscillation". After looking at the plot, it oscillated *more*
  with untuned gains. **Lesson:** read the figure before writing the sentence.

---

## v3: the LLM recipe (SFT → reward and cost models → PPO + KL)

**Motivation:** v2's two weaknesses mirror problems LLM RLHF already solved:
- Training from scratch under a strict constraint learns slowly. LLMs start from a capable SFT model.
- Policies exploit reward models. LLMs anchor to the SFT model with a KL penalty, and the literature adds pessimism or ensembles.
- Safe-RLHF (Dai et al. 2024) also *learns* the harm signal from human judgments instead of assuming it's given.

### What was added
1. **SFT-style reference** (`make_reference.py`):
   - Behavior-clone the v2 unconstrained PPO agent from 30k demonstration steps.
   - The clone is *capable but unsafe*: return ≈26 and about 12 goals per episode, but cost ≈60 (budget 25). That's the analogue of a helpful-but-unaligned SFT model.
   - It initializes the RLHF policy and is kept frozen as π_ref.
2. **KL anchor:**
   - Per-step reward `r_t − β (log π(a_t|s_t) − log π_ref(a_t|s_t))` with β = 0.1, the InstructGPT per-token form.
   - Its expectation is KL(π‖π_ref).
3. **Pessimistic ensembles:**
   - The policy optimizes `mean − κ·std` of the reward ensemble and `mean + κ·std` of the cost ensemble, with κ = 1.
   - *Subtlety:* Bradley-Terry is shift-invariant, so each reward member is standardized before measuring disagreement. Otherwise std would measure arbitrary offsets, not uncertainty.
   - The cost members aren't standardized, because the safety-classification term pins their zero point.
4. **Learned cost (Safe-RLHF):**
   - Every queried pair is answered twice (helpfulness and harmlessness), and every segment gets a binary safety label.
   - Cost loss: `−log σ(C_w − C_l) − log σ(z_w C_w) − log σ(z_l C_l)`, where z = +1 for unsafe and −1 for safe. A segment is predicted unsafe when C(σ) > 0.
   - Constraint: at most **2 predicted-unsafe 50-step segments per episode**.
   - *Calibration check:* the reference's 60 hazard steps correspond to about 5–6 truly unsafe segments, so 2 segments ≈ 22 hazard steps. That's close to v2's budget of 25, so the results are comparable.
5. **Engineering details that mattered:**
   - The observation normalizer is frozen at π_ref's statistics, so π and π_ref see identical inputs and the KL is meaningful.
   - One epoch of critic-only warm-up, so freshly initialized critics don't push the pretrained policy in random directions.

### Ablations (3 seeds each)
`v3_full` vs `v3_no_kl` (β=0) vs `v3_no_pess` (κ=0) vs `v3_true_cost` (constraint on the true hazard cost).
Runs: `runs/v3/v3.0/`, `runs/v3/v3.1/`. Figures: `results/v3/<version>/`. Launcher: `VERSION=v3.0|v3.1 scripts/run_v3.sh`.

### v3.0: the pretrained policy wouldn't move (500k steps)

| v3.0 (last 5 epochs, 3 seeds) | Task return | True cost |
|---|---|---|
| SFT reference | 26.3 | 60.4 |
| full / no_kl / no_pess / true_cost | 25.0 / 25.2 / 25.1 / 24.5 | 56.7 / 51.9 / 51.6 / 49.5 |

**No variant became safe, not even with the true cost and λ = 7.** Diagnosis from the logs (`results/v3/v3.0/`):
* KL to π_ref was only about 0.25 after 25 epochs (per-update KL ≈ 0.01, below PPO's 0.02 early-stop target), so the
  policy barely changed. The behavior-cloned policy is near-deterministic (action std ≈ 0.3 vs 0.6 for a fresh policy),
  so PPO's ratio clipping allows only tiny moves in the action mean, and there's little exploration toward safer actions.
* With the learned cost, λ rose roughly 6× more slowly than with the true cost: the constraint is measured in *segments*
  (errors of about 5), but λ's learning rate had been tuned for *hazard steps* (errors of about 30). **A unit mismatch.**
* On-policy reward-model correlation *rose* (0.08 → 0.13). No overoptimization, because the policy wasn't moving.

**Pilots before committing to a rerun (in scratch, not kept):**
1. Policy lr 1e-3, 20 PPO iterations, λ lr 0.1: λ rose faster, but cost didn't drop in 160k steps.
2. Adding an exploration reset (action std back to 0.6) and a warm-started λ: some cost reduction, but noisy (38–68).

No single knob fixed it, so I combined them and trained longer, accepting a full-length rerun.

### v3.1: plasticity restored; cost falls but capability falls faster (800k steps)

| v3.1 (last 5 epochs, 3 seeds) | Task return | True cost | λ | Pred. / true unsafe segments |
|---|---|---|---|---|
| full | 9.1 ± 2.3 | 43.3 ± 21.2 | 99 | 5.7 / 2.1 |
| no_kl | 6.4 ± 1.9 | 28.6 ± 10.1 | 92 | 4.8 / 1.5 |
| no_pess | 11.8 ± 6.4 | 41.2 ± 12.1 | 22 | 2.3 / 2.7 |
| true_cost | 6.8 ± 2.4 | 31.2 ± 3.9 | 39 | – / 1.7 |

**What I learned:**
1. **There's a trade-off curve, not a free lunch.** Cost falls from 60 toward the budget and is still trending down, but
   task return falls from 26 to 6–12. Every v3.1 variant sits *between* v2's endpoints. None meets the budget on average,
   so starting from a capable reference didn't beat training from scratch at this budget.
2. **Pessimism on the cost broke the Lagrangian.**
   - `mean + std` over-predicts unsafe segments (5.7 vs 2.1 true), so the budget of 2 is unreachable in the model's eyes.
   - λ climbs without bound (→ 99) and erodes the reward.
   - Without pessimism, the cost model is calibrated (2.3 vs 2.7), λ settles at 22, and the agent keeps the most capability.
   - Pessimism is a bias, and a fixed budget must account for it (or κ must be tuned, or the budget set on the pessimistic scale).
3. **Coarse safety labels saturate.** Under the learned cost, truly unsafe segments fall to about 2 per episode, but each
   holds 15–20 hazard steps, so the true cost stays around 40. I first wrote this up as "specification gaming", then
   checked the true-cost control: it shows the same concentration, so it's not gaming the label. Safer policies make fewer,
   longer incursions, and a binary per-clip label stops pushing once few clips are unsafe.
   **Lesson: check a striking claim against the control before keeping it.**
4. **KL to an unsafe reference costs safety.** Removing the KL anchor gives lower cost (29 vs 43) and lower return
   (6.4 vs 9.1). Anchoring preserves the reference's behavior, *including its unsafe habits*. That's the alignment-tax
   tension of LLM RLHF made visible.
5. **I could not show that KL or pessimism prevents overoptimization,** because overoptimization never appeared in v3
   (on-policy correlation rose in every variant). That claim needs a setting where it actually happens.

### What I would do next
* **Calibrate the cost side:** set the budget on the pessimistic scale or tune κ, and use per-step or graded safety labels instead of a binary per-clip label.
* **Use PID-Lagrangian with tuned gains, or a λ cap,** to prevent the runaway in pessimistic runs.
* **Train longer (2–5M steps) and add KL scheduling** (strong early anchor, decayed later), then measure the full Pareto front over β.
* **Create a regime where overoptimization happens** (e.g. the v2 summary-feature reward model) and test whether KL and pessimism suppress it. That's the cleanest experiment for the original v3 hypothesis.

---

## Lessons I'd present

1. **Claims must map to code and numbers.** v0 failed this, and everything after was built to pass it.
2. **Diagnose before you fix.** The reward-model plateau looked like overfitting, but offline experiments showed it was
   largely an observability ceiling.
3. **Report null results.** The v2 "fix" didn't significantly change final performance, and the HRL didn't help. Both are in the record.
4. **Accuracy on the labeled distribution isn't alignment.** Overoptimization only shows up on-policy, which is why v3
   adds KL anchoring and pessimism and tracks on-policy correlation.
5. **Safety and capability are a trade-off curve.** Starting from a capable reference (v3) moved *where* on the curve the
   agent starts, but at this budget it didn't produce a point that dominates training from scratch.
6. **Every ingredient interacts with the constraint.** Pessimism shifted the cost scale and broke the Lagrangian; the KL anchor
   pulled toward an unsafe reference; coarse safety labels saturated. Each LLM-RLHF component behaves differently once there's a hard budget.
7. **Check striking claims against controls.** The "specification gaming" story didn't survive the true-cost control.
