"""Figures for the README (PNG, light background), from the tables analysis.py wrote."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402

R = C.RESULTS
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
GREY = "#a3a29c"
SIZE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]  # ordinal blue
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False,
    "axes.titlelocation": "left", "axes.titlesize": 10.5, "axes.titlecolor": INK,
})
SIZES = ["70m", "160m", "410m", "1b", "1.4b", "2.8b"]
PARAMS = {"70m": 70e6, "160m": 160e6, "410m": 410e6, "1b": 1.0e9, "1.4b": 1.4e9, "2.8b": 2.8e9}


def save(fig, name):
    os.makedirs(C.FIGURES, exist_ok=True)
    fig.savefig(os.path.join(C.FIGURES, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def baseline(name, metric):
    b = pd.read_csv(os.path.join(R, "probe", "baselines.csv")).set_index("model")
    return b.loc[name, metric]


def fig_layers(layers):
    """Gain over the matching baseline at every layer, by relative depth."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, model, ref, title in [
            (axes[0], "N+H", "N", "Probe pooled with the n-gram baseline, gain over it"),
            (axes[1], "B+H", "B", "Probe pooled with spelling + frequency + meaning, gain over them")]:
        b = baseline(ref, "bal_acc")
        for size, col in zip(SIZES, SIZE_RAMP):
            d = layers[(layers.tag == "pythia-{}@main".format(size)) & (layers.pool == "last")
                       & (layers.model == model)].sort_values("layer")
            if d.empty:
                continue
            depth = d.layer / d.layer.max()
            ax.plot(depth, 100 * (d.bal_acc - b), color=col, lw=2, label=size)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("relative depth (0 = token embeddings, 1 = last layer)")
        ax.set_title(title)
    axes[0].set_ylabel("balanced accuracy gain (points)")
    axes[1].legend(title="Pythia", loc="lower right", ncol=2)
    save(fig, "fig1_layers.png")


def fig_scaling(head):
    """Headline gains (layer chosen inside training folds) against model size."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    series = [("N+H", "over the n-gram baseline", BLUE), ("S+H", "over both spelling models", ORANGE),
              ("B+H", "over spelling + frequency + meaning", AQUA)]
    for ax, (val, lo, hi, scale, ylab) in zip(axes, [
            ("acc_gain", "acc_lo", "acc_hi", 100, "balanced accuracy gain (points)"),
            ("ll_gain", "ll_lo", "ll_hi", 1, "balanced log loss gain (nats per word)")]):
        for model, label, col in series:
            d = head[(head.model == model) & (head.pool == "last") & head.tag.str.endswith("@main")
                     & head.tag.str.startswith("pythia")].copy()
            d["size"] = d.tag.str.extract(r"pythia-(.+)@")[0]
            d = d[d["size"].isin(SIZES)]
            d["p"] = d["size"].map(PARAMS)
            d = d.sort_values("p")
            if d.empty:
                continue
            ax.errorbar(d.p, scale * d[val], yerr=[scale * (d[val] - d[lo]), scale * (d[hi] - d[val])],
                        color=col, lw=2, marker="o", ms=6, capsize=0, elinewidth=1.2, label=label)
            o = head[(head.model == model) & (head.pool == "last") & head.tag.str.startswith("OLMo")]
            for i, (_, r) in enumerate(o.iterrows()):
                ax.errorbar(1.0e9 * (1.12 if "OLMo-2" in r.tag else 0.89), scale * r[val],
                            yerr=[[scale * (r[val] - r[lo])], [scale * (r[hi] - r[val])]],
                            color=col, marker="D", mfc=SURF, ms=6, lw=0, elinewidth=1.2)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("parameters")
        ax.set_ylabel(ylab)
    axes[0].legend(loc="upper left")
    axes[0].set_title("What the hidden states add, by model size")
    fig.text(0.5, -0.04, "circles: Pythia suite; hollow diamonds: OLMo 1 (left) and OLMo 2 (right), both 1B",
             ha="center", color=MUTED, fontsize=9)
    save(fig, "fig2_scaling.png")


def fig_classes(classes, tag):
    d = classes[(classes.tag == tag) & (classes.pool == "last")].sort_values("gain")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    y = np.arange(len(d))
    ax = axes[0]
    ax.hlines(y, d.recall_B * 100, d.recall_BH * 100, color=GREY, lw=2)
    ax.plot(d.recall_B * 100, y, "o", color=GREY, ms=7, label="spelling + frequency + meaning")
    ax.plot(d.recall_BH * 100, y, "o", color=BLUE, ms=7, label="... + hidden states")
    ax.set_yticks(y)
    ax.set_yticklabels(["{} ({})".format(c, n) for c, n in zip(d.cls, d.n)])
    ax.set_xlabel("recall: share of the class's words given the right donor (%)")
    ax.legend(loc="lower right")
    ax.set_title("Per donor class, {}".format(tag.replace("@main", "")))
    ax = axes[1]
    ax.barh(y, d.gain, color=BLUE, height=0.6)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("mean gain in log probability of the right donor (nats)")
    ax.set_title("Where the hidden states help")
    save(fig, "fig3_classes.png")


def fig_pairs(pairs, tag):
    d = pairs[(pairs.tag == tag) & (pairs.pool == "last")]
    order = [("N", "n-gram spelling"), ("S", "both spelling models"),
             ("B", "spelling + frequency + meaning"), ("H", "hidden states alone"),
             ("B+H", "everything + hidden states")]
    cols = [GREY, GREY, GREY, BLUE, AQUA]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, pair in zip(axes, ["Norse vs native", "Dutch vs native"]):
        e = d[d.pair == pair].set_index("model")
        y = np.arange(len(order))[::-1]
        for yy, (m, _), col in zip(y, order, cols):
            r = e.loc[m]
            ax.plot([r.lo, r.hi], [yy, yy], color=col, lw=2)
            ax.plot(r.auc, yy, "o", color=col, ms=7)
            ax.text(r.hi + 0.005, yy, "{:.2f}".format(r.auc), va="center", color=MUTED, fontsize=9)
        ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
        ax.set_yticks(y)
        ax.set_yticklabels([lab for _, lab in order])
        ax.set_xlabel("AUC (0.5 = chance)")
        ax.set_title(pair + ": loans that look native")
    save(fig, "fig4_pairs.png")


def fig_mentions(tag):
    path = os.path.join(R, "mentions_{}.csv".format(tag))
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    # remove what class and overall word frequency explain, as the regression does
    import statsmodels.formula.api as smf
    d["g"] = smf.ols("gain ~ log_count + label", data=d).fit().resid
    bins = [-0.5, 0.5, 1.5, 4.5, 16.5, 64.5, np.inf]
    labels = ["0", "1", "2-4", "5-16", "17-64", "65+"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for col, kind, name in [(BLUE, "true", "phrase naming the word's own donor"),
                            (GREY, "placebo", "phrase naming another donor (placebo)")]:
        b = pd.cut(d[kind], bins, labels=labels)
        s = d.groupby(b, observed=False).g.agg(["mean", "sem", "size"])
        x = np.arange(len(labels)) + (0.08 if kind == "placebo" else -0.08)
        ok = s["size"] >= 15
        ax.errorbar(x[ok], s["mean"][ok], yerr=1.96 * s["sem"][ok], color=col, lw=2, marker="o",
                    ms=6, elinewidth=1.2, label=name)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("times the Pile has the word within {} tokens of the phrase".format(C.WINDOW))
    ax.set_ylabel("gain from hidden states (nats),\nafter class and word frequency")
    ax.set_title("Does the gain track etymology talk in the training data? ({})".format(
        tag.replace("@main", "")))
    ax.legend(loc="upper left")
    save(fig, "fig5_mentions.png")


def fig_training(head):
    d = head[head.tag.str.startswith("pythia-410m@") & (head.pool == "last")].copy()
    if d.tag.nunique() < 3:
        return
    d["step"] = d.tag.str.extract(r"@(?:step)?(\w+)")[0].replace("main", "143000").astype(int)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for model, col, name in [("N+H", BLUE, "over the n-gram baseline"),
                             ("B+H", AQUA, "over spelling + frequency + meaning")]:
        e = d[d.model == model].sort_values("step")
        x = e.step.clip(lower=100)
        ax.errorbar(x, 100 * e.acc_gain, yerr=[100 * (e.acc_gain - e.acc_lo), 100 * (e.acc_hi - e.acc_gain)],
                    color=col, lw=2, marker="o", ms=6, elinewidth=1.2, label=name)
    ax.set_xscale("log")
    ticks = sorted(d.step.unique())
    ax.set_xticks([max(t, 100) for t in ticks])
    ax.set_xticklabels(["init" if t == 0 else "{:,}".format(t) for t in ticks])
    ax.minorticks_off()
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("training step (Pythia-410m; 143,000 is the released model, ~300B tokens)")
    ax.set_ylabel("balanced accuracy gain (points)")
    ax.set_title("When the extra signal appears during training")
    ax.legend(loc="upper left")
    save(fig, "fig6_training.png")


def main():
    layers = pd.read_csv(os.path.join(R, "layers.csv"))
    head = pd.read_csv(os.path.join(R, "headline.csv"))
    summary = json.load(open(os.path.join(R, "summary.json")))
    tag = summary["main_model"]
    fig_layers(layers)
    fig_scaling(head)
    fig_classes(pd.read_csv(os.path.join(R, "classes.csv")), tag)
    fig_pairs(pd.read_csv(os.path.join(R, "pairs.csv")), tag)
    fig_mentions(tag)
    fig_training(head)
    print("figures written to", C.FIGURES)


if __name__ == "__main__":
    main()
