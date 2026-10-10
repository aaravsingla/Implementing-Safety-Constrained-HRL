"""Aggregate runs/ into figures (results/*.png) and tables (results/summary.md, results/summary.csv).

Runs are grouped by name with the trailing `_s<seed>` removed; curves show mean +/- std over seeds.
"""

import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
RUNS, OUT = "runs/v2", "results/v2"
COST_LIMIT = 25.0

# fixed categorical order (colour follows the condition, never its rank)
CONDITIONS = {
    "ppolag_rm": ("PPO-Lag + learned RM (RLHF, ours)", "#2a78d6"),
    "ppolag_env": ("PPO-Lag + hand-coded reward", "#eb6834"),
    "ppolag_oracle": ("PPO-Lag + oracle reward (upper bound)", "#1baf7a"),
    "ppo_env": ("PPO, unconstrained", "#eda100"),
    "hrl_ppolag_rm": ("HRL Manager-Worker + PPO-Lag + RM", "#e87ba4"),
    "hrl_ppolag_env": ("HRL Manager-Worker + PPO-Lag + hand-coded", "#eb6834"),
    "v1_ppolag_env": ("PPO-Lag + hand-coded, cost adv. only centered (v1)", "#e34948"),
    "v1_ppolag_oracle": ("PPO-Lag + oracle, cost adv. only centered (v1)", "#4a3aa7"),
    "abl_activeq": ("RLHF, active queries (600k)", "#008300"),
    "abl_randomq": ("RLHF, random queries (600k)", "#eb6834"),
    "abl_budget500": ("RLHF, 500 labels (600k)", "#1baf7a"),
    "abl_noise0.3": ("RLHF, 30% label noise (600k)", "#eda100"),
    "abl_rmsummary": ("RLHF, lidar-summary RM features (600k)", "#e87ba4"),
    "abl_pid_env": ("PID-Lagrangian + hand-coded reward (600k)", "#4a3aa7"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2,
})


def load(root=RUNS, prefix=""):
    groups = {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        f = os.path.join(d, "progress.csv")
        if not os.path.exists(f) or os.path.getsize(f) == 0:
            continue
        name = os.path.basename(d)
        cond = prefix + re.sub(r"_s\d+$", "", name)
        df = pd.read_csv(f)
        final = None
        if os.path.exists(os.path.join(d, "final.json")):
            final = json.load(open(os.path.join(d, "final.json")))["eval"]
        groups.setdefault(cond, []).append((name, df, final))
    return groups


def curve(ax, groups, conds, metric, smooth=3):
    for c in conds:
        if c not in groups:
            continue
        dfs = [df for _, df, _ in groups[c] if metric in df]
        n = min(len(df) for df in dfs)
        y = np.stack([df[metric].to_numpy(dtype=float)[:n] for df in dfs])
        y = pd.DataFrame(y.T).rolling(smooth, min_periods=1).mean().to_numpy().T
        x = dfs[0]["steps"].to_numpy()[:n] / 1e6
        m, s = y.mean(0), y.std(0)
        label, color = CONDITIONS.get(c, (c, MUTED))
        ax.plot(x, m, color=color, label=f"{label} (n={len(dfs)})")
        if len(dfs) > 1:
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("environment steps (M)")


def fig_main(groups, conds, fname, title):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (metric, lab) in zip(axes, [("ep_oracle_ret", "supervisor (oracle) return ↑"),
                                        ("ep_ret", "hand-coded task return ↑"),
                                        ("ep_cost", "episode cost ↓"),
                                        ("near_miss_rate", "near-miss rate ↓")]):
        curve(ax, groups, conds, metric)
        ax.set_title(lab, loc="left", color=INK)
        if metric == "ep_cost":
            ax.axhline(COST_LIMIT, color=INK, ls="--", lw=1)
            ax.text(0.02, COST_LIMIT, " cost limit d=25", va="bottom", color=MUTED, transform=ax.get_yaxis_transform())
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(4, len(l)), bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_rm(groups):
    conds = [c for c in ["ppolag_rm", "abl_activeq", "abl_randomq", "abl_budget500", "abl_noise0.3", "abl_rmsummary"] if c in groups]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (metric, lab) in zip(axes, [("rm_val_acc", "held-out preference accuracy ↑"),
                                        ("rm_corr", "corr(learned r̂, oracle r) per step ↑"),
                                        ("n_labels", "preference labels used")]):
        curve(ax, groups, conds, metric, smooth=1 if metric == "n_labels" else 3)
        ax.set_title(lab, loc="left", color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Reward model diagnostics", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "reward_model.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_lambda(groups):
    conds = [c for c in ["ppolag_env", "abl_pid_env", "ppolag_rm"] if c in groups]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    curve(axes[0], groups, conds, "lambda", smooth=1)
    axes[0].set_title("Lagrange multiplier λ", loc="left", color=INK)
    curve(axes[1], groups, conds, "ep_cost", smooth=1)
    axes[1].axhline(COST_LIMIT, color=INK, ls="--", lw=1)
    axes[1].set_title("episode cost (unsmoothed) ↓", loc="left", color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "lagrangian.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def table(groups):
    rows = []
    metrics = ["ep_oracle_ret", "ep_ret", "ep_cost", "near_miss_rate", "goals_per_ep"]
    for c in CONDITIONS:
        if c not in groups:
            continue
        runs = groups[c]
        last = {m: [df[m].tail(5).mean() for _, df, _ in runs] for m in metrics}
        evals = [f for _, _, f in runs if f is not None]
        row = {"condition": CONDITIONS[c][0], "seeds": len(runs), "epochs": int(min(len(df) for _, df, _ in runs))}
        for m in metrics:
            row[f"train_{m}"] = f"{np.mean(last[m]):.2f} ± {np.std(last[m]):.2f}"
        if evals:
            for m in metrics:
                v = [e[m] for e in evals]
                row[f"eval_{m}"] = f"{np.mean(v):.2f} ± {np.std(v):.2f}"
        if "rm_val_acc" in runs[0][1]:
            v = [df["rm_val_acc"].iloc[-1] for _, df, _ in runs]
            r = [df["rm_corr"].tail(5).mean() for _, df, _ in runs]
            row["rm_val_acc"] = f"{np.mean(v):.3f}"
            row["rm_corr"] = f"{np.mean(r):.3f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("Training metrics: mean of the last 5 epochs (100k steps), mean ± std over seeds.\n")
        f.write("Eval: 10 deterministic episodes on held-out layouts per seed.\n\n")
        f.write(df.to_markdown(index=False))
    print(df.to_string(index=False))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    g = load()
    g.update(load("runs/v1", "v1_"))
    fig_main(g, ["ppolag_rm", "ppolag_env", "ppolag_oracle", "ppo_env"], "main_comparison.png",
             "Safe RL from preferences on SafetyPointGoal1 (mean ± std over seeds)")
    fig_main(g, ["ppolag_rm", "hrl_ppolag_rm", "hrl_ppolag_env"], "hrl_comparison.png",
             "Flat vs hierarchical (Manager-Worker) agents")
    fig_main(g, ["ppolag_env", "v1_ppolag_env", "ppolag_oracle", "v1_ppolag_oracle"], "cost_adv_fix.png",
             "Effect of standardizing the cost advantage (v1 = centered only)")
    fig_rm(g)
    fig_lambda(g)
    table(g)
