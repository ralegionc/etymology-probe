"""Print every number the README quotes, from the saved results (python report.py)."""
import json
import os

import numpy as np
import pandas as pd

import config as C

R = C.RESULTS


def pct(x):
    return "{:.1f}".format(100 * x)


def gain_line(r, acc=True):
    if acc:
        return "{:+.1f} points [{:+.1f}, {:+.1f}]".format(100 * r.acc_gain, 100 * r.acc_lo, 100 * r.acc_hi)
    return "{:+.3f} nats [{:+.3f}, {:+.3f}]".format(r.ll_gain, r.ll_lo, r.ll_hi)


def main():
    s = json.load(open(os.path.join(R, "summary.json")))
    build = json.load(open(os.path.join(R, "words_build.json")))
    head = pd.read_csv(os.path.join(R, "headline.csv"))
    base = pd.read_csv(os.path.join(R, "probe", "baselines.csv")).set_index("model")
    main_tag = s["main_model"]

    print("## Data")
    for k, v in build.items():
        print("-", k, ":", v)

    print("\n## Baselines (balanced accuracy, balanced log loss)")
    for m in ["prior", "spell", "cnn", "freq", "sem", "N", "S", "SF", "SS", "B"]:
        print("- {:6s} {}%  {:.3f}".format(m, pct(base.loc[m, "bal_acc"]), base.loc[m, "bal_logloss"]))

    print("\n## Headline, layer chosen inside each training fold")
    h = head[head.pool == "last"].copy()
    for tag, g in h.groupby("tag", sort=False):
        g = g.set_index("model")
        print("\n### " + tag)
        for m in ["H", "N+H", "S+H", "B+H"]:
            r = g.loc[m]
            print("- {:4s} vs {}: {}% ({}; {}), layers {}".format(
                m, r.vs, pct(r.bal_acc), gain_line(r), gain_line(r, False), r.layers))

    mean = head[head.pool == "mean"]
    if len(mean):
        print("\n## Mean pooling instead of last token")
        for _, r in mean[mean.model.isin(["H", "B+H"])].iterrows():
            print("- {} {}: {}% ({})".format(r.tag, r.model, pct(r.bal_acc), gain_line(r)))

    print("\n## Per class, " + main_tag)
    c = pd.read_csv(os.path.join(R, "classes.csv"))
    c = c[(c.tag == main_tag) & (c.pool == "last")].sort_values("gain", ascending=False)
    for _, r in c.iterrows():
        print("- {:9s} n={:5d} recall {}% -> {}%  gain {:+.3f} nats".format(
            r.cls, r.n, pct(r.recall_B), pct(r.recall_BH), r.gain))

    print("\n## Pairs (AUC [95% CI]), " + main_tag)
    p = pd.read_csv(os.path.join(R, "pairs.csv"))
    p = p[(p.tag == main_tag) & (p.pool == "last")]
    for pair, g in p.groupby("pair"):
        print("- " + pair + ": " + ", ".join("{} {:.3f} [{:.3f}, {:.3f}]".format(
            r.model, r.auc, r.lo, r.hi) for _, r in g.iterrows()))
    for pair, g in pd.read_csv(os.path.join(R, "pairs.csv")).query("pool == 'last'").groupby("pair"):
        g = g.pivot(index="tag", columns="model", values="auc")
        print("  " + pair + " by model:\n" + g.round(3).to_string())

    print("\n## Disguised loans (spelling says native), " + main_tag)
    d = s["disguised"]
    print("- n={} by class {}".format(d["n"], d["by_class"]))
    print("- right donor: B {}%, B+H {}%, mean gain {:+.3f} nats".format(
        pct(d["B_correct"]), pct(d["BH_correct"]), d["mean_gain"]))
    dis = pd.read_csv(os.path.join(R, "disguised.csv"))
    won = dis[(dis.BH_top == dis.label) & (dis.B_top != dis.label)].sort_values("zipf", ascending=False)
    lost = dis[(dis.BH_top != dis.label) & (dis.B_top == dis.label)]
    print("- recovered by the hidden states ({}): {}".format(
        len(won), ", ".join("{} ({})".format(a, b) for a, b in zip(won.word[:40], won.label[:40]))))
    print("- lost ({}): {}".format(len(lost), ", ".join(lost.word[:20])))
    for wd in ["they", "them", "sky", "egg", "knife", "take", "get", "give", "window", "husband",
               "law", "wrong", "ugly", "boss", "cookie", "yacht", "skipper"]:
        r = pd.read_csv(os.path.join(R, "words_{}.csv".format(main_tag))).set_index("word")
        if wd in r.index:
            x = r.loc[wd]
            print("  {:8s} {:7s} B->{:8s} B+H->{:8s} logp {:.2f} -> {:.2f}".format(
                wd, x.label, x.top_B, x.top_BH, x.logp_B, x.logp_BH))

    print("\n## Explicit labels only")
    e = pd.read_csv(os.path.join(R, "sensitivity_explicit.csv"))
    print("- subset:", s["explicit_subset"])
    for _, r in e.iterrows():
        print("- {} {} vs {}: {}".format(r.tag, r.model, r.vs, gain_line(r)))

    print("\n## Mentions")
    for k, m in s.get("mentions", {}).items():
        terms = {t: v for t, v in m["true+placebo"].items() if t != "n"}
        print("- {} (n={}): ".format(k, m["true+placebo"]["n"]) + ", ".join(
            "{} {:+.4f} (se {:.4f}, p {:.3g})".format(t, v["coef"], v["se"], v["p"])
            for t, v in terms.items()))
    path = os.path.join(R, "mentions_{}.csv".format(main_tag))
    if os.path.exists(path):
        m = pd.read_csv(path)
        print("- words with counts: {} ; share with any true co-mention {:.1%}; median true {}".format(
            len(m), (m["true"] > 0).mean(), m["true"].median()))
        print("- approx (subsampled) word counts: {:.1%}".format(m.true_approx.mean()))
        print(m.groupby("label")["true"].agg(["median", "mean", "size"]).round(1).to_string())

    print("\n## Behaviour (asked directly, calibrated)")
    for k, v in s.get("behave", {}).items():
        print("- {}: {}".format(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                                    for kk, vv in v.items()}))


if __name__ == "__main__":
    main()
