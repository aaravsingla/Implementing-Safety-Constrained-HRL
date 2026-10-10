"""Figures for v3 (results/v3/) and the cross-version comparison table (results/journey_table.md)."""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot as P

VERSION = sys.argv[1] if len(sys.argv) > 1 else "v3.1"
OUT = f"results/v3/{VERSION}"
P.CONDITIONS.update({
    "v3_full": ("v3: SFT ref + KL + pessimistic RM + learned cost (full)", "#2a78d6"),
    "v3_no_kl": ("v3 without KL anchor (β=0)", "#eb6834"),
    "v3_no_pess": ("v3 without pessimism (κ=0)", "#1baf7a"),
    "v3_true_cost": ("v3 with true cost instead of learned cost", "#eda100"),
})
V3 = ["v3_full", "v3_no_kl", "v3_no_pess", "v3_true_cost"]


def reference_eval():
    vals = [json.load(open(f"runs/v3/reference_s{s}/final.json"))["eval"] for s in range(3)
            if os.path.exists(f"runs/v3/reference_s{s}/final.json")]
    return {k: (np.mean([v[k] for v in vals]), np.std([v[k] for v in vals])) for k in vals[0]} if vals else None


def fig_main(g, ref):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    panels = [("ep_ret", "hand-coded task return ↑", "ep_ret"), ("ep_cost", "true episode cost ↓", "ep_cost"),
              ("ep_oracle_ret", "supervisor (oracle) return ↑", "ep_oracle_ret"), ("kl_ref", "KL(π ‖ π_ref) per step", None)]
    for ax, (metric, title, refkey) in zip(axes, panels):
        P.curve(ax, g, V3, metric)
        ax.set_title(title, loc="left", color=P.INK)
        if refkey and ref:
            ax.axhline(ref[refkey][0], color=P.MUTED, ls=":", lw=1.2)
            ax.text(0.98, ref[refkey][0], "SFT reference ", ha="right", va="bottom", color=P.MUTED, fontsize=8,
                    transform=ax.get_yaxis_transform())
        if metric == "ep_cost":
            ax.axhline(P.COST_LIMIT, color=P.INK, ls="--", lw=1)
            ax.text(0.02, P.COST_LIMIT, " v2 budget d=25", va="bottom", color=P.MUTED, transform=ax.get_yaxis_transform())
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.14))
    fig.suptitle(f"{VERSION}: RLHF fine-tuning from an SFT reference (mean ± std over 3 seeds)", x=0.01, ha="left", color=P.INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v3_main.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_models(g):
    if "v3_full" not in g:
        return
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    panels = [("rm_corr", "on-policy corr(r̂, true r) ↑", V3),
              ("rm_val_acc", "reward-model held-out pref. acc. ↑", V3),
              ("cm_corr", "on-policy corr(ĉ, true cost) ↑", ["v3_full", "v3_no_kl", "v3_no_pess"]),
              ("unsafe_segs_pred", "predicted vs true unsafe segments / ep", ["v3_full"])]
    for ax, (metric, title, conds) in zip(axes, panels):
        P.curve(ax, g, conds, metric)
        ax.set_title(title, loc="left", color=P.INK)
    P.curve(axes[3], {"true": g["v3_full"]}, ["true"], "unsafe_segs_true")
    axes[3].lines[-1].set_color(P.INK); axes[3].lines[-1].set_linestyle("--")
    axes[3].lines[-1].set_label("true (full)")
    axes[3].lines[0].set_label("predicted by cost model (full)")
    axes[3].axhline(2.0, color=P.MUTED, ls=":", lw=1)
    axes[3].legend(loc="best", fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.14))
    fig.suptitle(f"{VERSION} reward/cost model diagnostics", x=0.01, ha="left", color=P.INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v3_models.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def journey_table(g, ref):
    metrics = ["ep_ret", "ep_cost", "ep_oracle_ret", "near_miss_rate", "goals_per_ep"]

    def agg(cond):
        runs = g[cond]
        last = {m: [df[m].tail(5).mean() for _, df, _ in runs] for m in metrics}
        return {m: f"{np.mean(v):.1f} ± {np.std(v):.1f}" if m != "near_miss_rate" else f"{np.mean(v):.2f}"
                for m, v in last.items()}, len(runs), int(runs[0][1]["steps"].iloc[-1])

    rows = []
    spec = [("v1", "v1_ppolag_rm", "PPO-Lag + learned RM, cost advantage only centered"),
            ("v1", "v1_ppolag_env", "PPO-Lag + hand-coded reward (v1)"),
            ("v2", "ppo_env", "PPO unconstrained (capable, unsafe)"),
            ("v2", "ppolag_env", "PPO-Lag + hand-coded reward"),
            ("v2", "ppolag_rm", "PPO-Lag + learned RM (RLHF from scratch)"),
            ("v2", "hrl_ppolag_rm", "HRL Manager-Worker + PPO-Lag + learned RM")]
    for ver, cond, label in spec:
        if cond in g:
            m, n, steps = agg(cond)
            rows.append({"version": ver, "agent": label, "seeds": n, "RL steps": f"{steps/1e6:.1f}M", **m})
    if ref:
        rows.append({"version": "v3", "agent": "SFT reference (behavior cloning, eval)", "seeds": 3, "RL steps": "0 (+30k demos)",
                     **{m: (f"{ref[m][0]:.1f} ± {ref[m][1]:.1f}" if m != "near_miss_rate" else f"{ref[m][0]:.2f}") for m in metrics}})
    for ver in ("v3.0", "v3.1"):
        for cond in V3:
            key = f"{ver}_{cond}"
            if key in g:
                m, n, steps = agg(key)
                rows.append({"version": ver, "agent": P.CONDITIONS[cond][0].replace("v3: ", "").replace("v3 ", ""),
                             "seeds": n, "RL steps": f"{steps/1e6:.1f}M", **m})
    df = pd.DataFrame(rows).rename(columns={"ep_ret": "task return", "ep_cost": "true cost", "ep_oracle_ret": "supervisor return",
                                            "near_miss_rate": "near-miss", "goals_per_ep": "goals/ep"})
    with open("results/journey_table.md", "w") as f:
        f.write("Mean ± std over seeds of the last 5 training epochs (SFT reference: 10 deterministic eval episodes per seed).\n")
        f.write("Cost budget: 25 hazard steps / episode (v1, v2, v3 true-cost); 2 unsafe 50-step segments / episode as judged by the learned cost model (other v3 rows; ≈22 hazard steps).\n"
                "v3 rows start from the SFT reference, which itself was cloned from 1M steps of unconstrained PPO.\n\n")
        f.write(df.to_markdown(index=False))
    print(df.to_string(index=False))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    allg = P.load("runs/v2")
    allg.update(P.load("runs/v1", "v1_"))
    for ver in ("v3.0", "v3.1"):
        allg.update(P.load(f"runs/v3/{ver}", f"{ver}_"))
    ref = reference_eval()
    g = P.load(f"runs/v3/{VERSION}")
    if g:
        fig_main(g, ref)
        fig_models(g)
    journey_table(allg, ref)
