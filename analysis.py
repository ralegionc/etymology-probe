"""Every number the README reports, from the saved out-of-fold probabilities.

For each probed model and checkpoint:

  layers     per-layer scores of the probe alone (H) and of the probe pooled with
             the n-gram baseline (N+H), with spelling (S+H) and with everything (B+H)
  headline   the same stacks with the layer chosen inside each training fold, so the
             layer choice never sees the words it is scored on, and the gain over the
             matching baseline with a 95% bootstrap interval
  classes    per-class recall and mean log probability of the right donor, B vs B+H
  pairs      Norse vs native and Dutch vs native: can the scores tell the two apart
             (AUC, words of the two classes only)?
  disguised  loans the spelling models think are native: what B and B+H make of them
  mentions   does the per-word gain track how often the Pile puts the word next to
             a phrase naming its donor ("from Old Norse")? Placebo and generic
             co-mentions are the controls.

Writes results/*.csv and results/summary.json.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

import config as C
import probe as P

OUT = C.RESULTS
B_DRAWS = 1000
PAIRS = [("Norse", "native"), ("Dutch", "native")]
MAIN = "pythia-2.8b@main"       # the model the per-word analyses use
SIZES = {"70m": 70e6, "160m": 160e6, "410m": 410e6, "1b": 1.0e9, "1.4b": 1.4e9, "2.8b": 2.8e9}


# ---------------------------------------------------------------- stacking

def pool_fold(parts, y, tr):
    a, b = P.fit_pool(parts[:, tr], y[tr], P.class_weights(y[tr]))
    return a, b


def nll_weighted(Q, y, w):
    p = np.clip(Q[np.arange(len(y)), y], P.EPS, 1)
    return float((w * -np.log(p)).sum() / w.sum())


def stacked_layers(fixed, H, y, fold):
    """For every layer l: the out-of-fold pool of fixed + H[l], and each fold's
    training loss (used to pick a layer without looking at the test fold)."""
    Lf = np.log(np.clip(np.array(fixed), P.EPS, 1)) if fixed else np.zeros((0,) + H.shape[1:])
    LH = np.log(np.clip(H, P.EPS, 1))
    n_layers = H.shape[0]
    out = np.zeros(H.shape)
    train_loss = np.zeros((P.K_FOLDS, n_layers))
    for l in range(n_layers):
        parts = np.concatenate([Lf, LH[l:l + 1]])
        for f in range(P.K_FOLDS):
            tr, te = fold != f, fold == f
            a, b = pool_fold(parts, y, tr)
            out[l, te] = softmax(np.tensordot(a, parts[:, te], 1) + b, axis=1)
            Qtr = softmax(np.tensordot(a, parts[:, tr], 1) + b, axis=1)
            train_loss[f, l] = nll_weighted(Qtr, y[tr], P.class_weights(y[tr]))
    return out, train_loss


def select(stacked, train_loss, fold):
    """Each fold's test words get the layer that did best on the other folds."""
    Q = np.zeros(stacked.shape[1:])
    chosen = []
    for f in range(P.K_FOLDS):
        l = int(np.argmin(train_loss[f]))
        Q[fold == f] = stacked[l, fold == f]
        chosen.append(l)
    return Q, chosen


# ---------------------------------------------------------------- statistics

def boot_gain(y, Q1, Q0, draws=B_DRAWS, seed=0):
    """Bootstrap (words resampled within class) of the gain of Q1 over Q0 in balanced
    accuracy and balanced log loss (lower is better, so the gain is Q0 minus Q1)."""
    rng = np.random.default_rng(seed)
    hit1 = (Q1.argmax(1) == y).astype(float)
    hit0 = (Q0.argmax(1) == y).astype(float)
    l1 = -np.log(np.clip(Q1[np.arange(len(y)), y], P.EPS, 1))
    l0 = -np.log(np.clip(Q0[np.arange(len(y)), y], P.EPS, 1))
    acc, ll = np.zeros(draws), np.zeros(draws)
    classes = np.unique(y)
    for c in classes:
        i = np.flatnonzero(y == c)
        s = rng.integers(0, len(i), (draws, len(i)))
        acc += (hit1[i][s] - hit0[i][s]).mean(1)
        ll += (l0[i][s] - l1[i][s]).mean(1)
    acc /= len(classes)
    ll /= len(classes)
    point_acc = np.mean([(hit1[y == c] - hit0[y == c]).mean() for c in classes])
    point_ll = np.mean([(l0[y == c] - l1[y == c]).mean() for c in classes])
    return dict(acc_gain=point_acc, acc_lo=np.percentile(acc, 2.5), acc_hi=np.percentile(acc, 97.5),
                ll_gain=point_ll, ll_lo=np.percentile(ll, 2.5), ll_hi=np.percentile(ll, 97.5))


def pair_auc(y, Q, a, b, classes, draws=B_DRAWS, seed=0):
    ia, ib = classes.index(a), classes.index(b)
    m = (y == ia) | (y == ib)
    score = np.log(np.clip(Q[m, ia], P.EPS, 1)) - np.log(np.clip(Q[m, ib], P.EPS, 1))
    t = (y[m] == ia).astype(int)
    auc = roc_auc_score(t, score)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
    boots = []
    for _ in range(draws // 5):
        s = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        boots.append(roc_auc_score(t[s], score[s]))
    return auc, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


# ---------------------------------------------------------------- per model

def tags(pool="last"):
    paths = sorted(glob.glob(os.path.join(P.PROBE_DIR, "*_{}_oof.npz".format(pool))))
    return [os.path.basename(p)[:-len("_{}_oof.npz".format(pool))] for p in paths]


def analyse_model(tag, pool, y, fold, base, classes):
    """Cached on the probe and baseline files' modification times."""
    src = os.path.join(P.PROBE_DIR, "{}_{}_oof.npz".format(tag, pool))
    key = "{:.0f}-{:.0f}".format(os.path.getmtime(src),
                                 os.path.getmtime(os.path.join(P.PROBE_DIR, "baselines.npz")))
    cache = os.path.join(OUT, "cache", "{}_{}.npz".format(tag, pool))
    if os.path.exists(cache):
        c = np.load(cache, allow_pickle=True)
        if str(c["key"]) == key:
            return (pd.DataFrame(c["layers"].item()), pd.DataFrame(c["head"].item()),
                    {k: c["pick_" + k] for k in ("H", "N+H", "S+H", "B+H")})
    layers, head, picks = _analyse_model(src, tag, pool, y, fold, base, classes)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, key=key, layers=layers.to_dict("list"), head=head.to_dict("list"),
             **{"pick_" + k: v for k, v in picks.items()})
    return layers, head, picks


def _analyse_model(src, tag, pool, y, fold, base, classes):
    H = np.load(src)["H"].astype(float)
    rows, picks = [], {}
    stacks = {"H": [], "N+H": [base["spell"]], "S+H": [base[p] for p in P.SPELL],
              "B+H": [base[p] for p in P.BASE]}
    ref = {"H": "N", "N+H": "N", "S+H": "S", "B+H": "B"}
    for name, fixed in stacks.items():
        if name == "H":
            st = H
            tl = np.array([[P.bal_logloss(y[fold != f], H[l, fold != f]) for l in range(len(H))]
                           for f in range(P.K_FOLDS)])
        else:
            st, tl = stacked_layers(fixed, H, y, fold)
        for l in range(len(H)):
            rows.append(dict(tag=tag, pool=pool, layer=l, model=name, **P.metrics(y, st[l])))
        Q, chosen = select(st, tl, fold)
        picks[name] = (Q, chosen)
    layers = pd.DataFrame(rows)
    head = []
    for name, (Q, chosen) in picks.items():
        r = dict(tag=tag, pool=pool, model=name, vs=ref[name], layers=str(chosen),
                 **P.metrics(y, Q))
        r.update(boot_gain(y, Q, base[ref[name]]))
        head.append(r)
    return layers, pd.DataFrame(head), {k: v[0] for k, v in picks.items()}


def per_class(y, classes, Q0, Q1):
    rows = []
    for i, c in enumerate(classes):
        m = y == i
        lp0 = np.log(np.clip(Q0[m, i], P.EPS, 1))
        lp1 = np.log(np.clip(Q1[m, i], P.EPS, 1))
        rows.append(dict(cls=c, n=int(m.sum()),
                         recall_B=(Q0[m].argmax(1) == i).mean(), recall_BH=(Q1[m].argmax(1) == i).mean(),
                         logp_B=lp0.mean(), logp_BH=lp1.mean(), gain=(lp1 - lp0).mean()))
    return pd.DataFrame(rows)


def disguised(w, y, classes, S, Bq, BHq):
    """Loans whose spelling says native: the spelling stack's top class is native."""
    nat = classes.index(C.NATIVE)
    m = (y != nat) & (S.argmax(1) == nat)
    d = w[m].copy()
    d["S_top"] = C.NATIVE
    d["B_top"] = [classes[i] for i in Bq[m].argmax(1)]
    d["BH_top"] = [classes[i] for i in BHq[m].argmax(1)]
    idx = np.arange(len(y))[m]
    d["logp_B"] = np.log(np.clip(Bq[idx, y[m]], P.EPS, 1))
    d["logp_BH"] = np.log(np.clip(BHq[idx, y[m]], P.EPS, 1))
    d["gain"] = d.logp_BH - d.logp_B
    return d


# ---------------------------------------------------------------- mentions

def mention_table(w, y, Bq, BHq):
    path = os.path.join(C.DATA, "counts_pythia.parquet")
    if not os.path.exists(path):
        return None
    cnt = pd.read_parquet(path)
    d = w[["word", "label", "zipf", "kind"]].copy()
    idx = np.arange(len(y))
    d["gain"] = (np.log(np.clip(BHq[idx, y], P.EPS, 1)) - np.log(np.clip(Bq[idx, y], P.EPS, 1)))
    d["logp_B"] = np.log(np.clip(Bq[idx, y], P.EPS, 1))
    d = d.merge(cnt.drop(columns=["label"]), on="word", how="inner")
    d = d.dropna(subset=["count", "true", "placebo"])
    d = d[d["count"] > 0]
    d["log_count"] = np.log10(d["count"])
    for k in ("true", "placebo", "generic"):
        if k in d:
            d["log_" + k] = np.log10(d[k] + 1)
    # co-mentions chance alone would give: word and phrase independently placed
    d["exp_true"] = d["count"] * d.total_true * 2 * C.WINDOW / 3.83e11
    return d


def mention_models(d):
    """OLS of the per-word gain on co-mention counts, controls and class effects."""
    out = {}
    specs = {
        "true": "gain ~ log_true + log_count + label",
        "true+placebo": "gain ~ log_true + log_placebo + log_count + label",
        "true+placebo+generic": "gain ~ log_true + log_placebo + log_generic + log_count + label",
        "true+placebo+B": "gain ~ log_true + log_placebo + log_count + logp_B + label",
    }
    for name, f in specs.items():
        if "generic" in f and d.log_generic.isna().any():
            dd = d.dropna(subset=["log_generic"])
        else:
            dd = d
        m = smf.ols(f, data=dd).fit(cov_type="HC3")
        out[name] = dict(n=int(m.nobs), **{k: dict(coef=float(m.params[k]), se=float(m.bse[k]),
                                                    p=float(m.pvalues[k]))
                                            for k in m.params.index if not k.startswith("label[")})
    return out


# ---------------------------------------------------------------- behaviour

def behave_summary(w, y, classes, preds):
    """Asked directly (behave.py), does the model name the right donor? Compared with
    the probe on the same words."""
    out = {}
    for path in sorted(glob.glob(os.path.join(OUT, "behave_*.parquet"))):
        tag = os.path.basename(path)[len("behave_"):-len(".parquet")]
        b = pd.read_parquet(path)
        raw = b[classes].values
        cal = raw - raw.mean(0)
        yb = b.label.map({c: i for i, c in enumerate(classes)}).values
        idx = w.reset_index().set_index("word").loc[b.word, "index"].values
        r = dict(n=len(b))
        for name, S in (("raw", raw), ("calibrated", cal)):
            hit = S.argmax(1) == yb
            r[name + "_bal_acc"] = float(pd.Series(hit).groupby(yb).mean().mean())
        Q = softmax(cal, axis=1)
        for a, bb in PAIRS:
            r["auc " + a + " vs " + bb] = pair_auc(yb, Q, a, bb, classes)[0]
        if (tag, "last") in preds:
            H = preds[(tag, "last")]["H"][idx]
            r["probe_bal_acc_same_words"] = float(
                pd.Series(H.argmax(1) == yb).groupby(yb).mean().mean())
            r["per_class_calibrated"] = pd.Series(cal.argmax(1) == yb).groupby(
                b.label.values).mean().round(3).to_dict()
        out[tag] = r
    return out


# ---------------------------------------------------------------- main

def main():
    w, y, classes = P.load_words()
    fold = P.folds(w, y)
    base = dict(np.load(os.path.join(P.PROBE_DIR, "baselines.npz")))
    classes = list(classes)
    summary = {"n_words": len(w), "classes": w.label.value_counts().to_dict(),
               "baselines": pd.read_csv(os.path.join(P.PROBE_DIR, "baselines.csv"))
                              .set_index("model").round(4).to_dict("index")}
    all_layers, all_head, preds = [], [], {}
    for pool in ("last", "mean"):
        for tag in tags(pool):
            layers, head, picks = analyse_model(tag, pool, y, fold, base, classes)
            all_layers.append(layers)
            all_head.append(head)
            preds[(tag, pool)] = picks
            print(head[["tag", "pool", "model", "bal_acc", "bal_logloss", "acc_gain", "ll_gain",
                        "ll_lo", "ll_hi"]].round(3).to_string(index=False), flush=True)
    layers = pd.concat(all_layers)
    head = pd.concat(all_head)
    layers.to_csv(os.path.join(OUT, "layers.csv"), index=False)
    head.to_csv(os.path.join(OUT, "headline.csv"), index=False)

    # pairs and per-class results for every final model
    pair_rows, class_rows = [], []
    for (tag, pool), picks in preds.items():
        cand = {"N": base["N"], "S": base["S"], "B": base["B"], "H": picks["H"],
                "B+H": picks["B+H"]}
        for a, b in PAIRS:
            for name, Q in cand.items():
                auc, lo, hi = pair_auc(y, Q, a, b, classes)
                pair_rows.append(dict(tag=tag, pool=pool, pair="{} vs {}".format(a, b),
                                      model=name, auc=auc, lo=lo, hi=hi))
        pc = per_class(y, classes, base["B"], picks["B+H"])
        pc.insert(0, "tag", tag)
        pc.insert(1, "pool", pool)
        class_rows.append(pc)
    pd.DataFrame(pair_rows).to_csv(os.path.join(OUT, "pairs.csv"), index=False)
    pd.concat(class_rows).to_csv(os.path.join(OUT, "classes.csv"), index=False)

    main_tag = MAIN if (MAIN, "last") in preds else sorted(t for t, p in preds if p == "last")[-1]
    picks = preds[(main_tag, "last")]
    summary["main_model"] = main_tag
    dis = disguised(w, y, classes, base["S"], base["B"], picks["B+H"])
    dis.to_csv(os.path.join(OUT, "disguised.csv"), index=False)
    summary["disguised"] = dict(
        n=len(dis), by_class=dis.label.value_counts().to_dict(),
        B_correct=float((dis.B_top == dis.label).mean()),
        BH_correct=float((dis.BH_top == dis.label).mean()),
        mean_gain=float(dis.gain.mean()))

    # per-word gains for the main model, and mention regressions for every Pythia model
    words_out = w[["word", "label", "kind", "zipf"]].copy()
    idx = np.arange(len(y))
    for name in ("N", "S", "B"):
        words_out["logp_" + name] = np.log(np.clip(base[name][idx, y], P.EPS, 1))
    words_out["logp_BH"] = np.log(np.clip(picks["B+H"][idx, y], P.EPS, 1))
    words_out["logp_H"] = np.log(np.clip(picks["H"][idx, y], P.EPS, 1))
    words_out["top_B"] = [classes[i] for i in base["B"].argmax(1)]
    words_out["top_BH"] = [classes[i] for i in picks["B+H"].argmax(1)]
    words_out.to_csv(os.path.join(OUT, "words_{}.csv".format(main_tag)), index=False)

    ment = {}
    for (tag, pool), pk in preds.items():
        if pool != "last" or not tag.startswith("pythia"):
            continue
        for ref, stack_name in (("B", "B+H"), ("S", "S+H")):
            d = mention_table(w, y, base[ref], pk[stack_name])
            if d is None or len(d) < 200:
                continue
            ment["{} {} vs {}".format(tag, stack_name, ref)] = mention_models(d)
            loans = d[d.label != C.NATIVE]
            ment["{} {} vs {}, loans only".format(tag, stack_name, ref)] = mention_models(loans)
            # words rare enough that the engine counted co-mentions exactly
            exact = d[~d.true_approx.astype(bool) & ~d.placebo_approx.astype(bool)]
            ment["{} {} vs {}, exact counts".format(tag, stack_name, ref)] = mention_models(exact)
            if tag == main_tag and ref == "B":
                d.to_csv(os.path.join(OUT, "mentions_{}.csv".format(tag)), index=False)
    summary["mentions"] = ment

    # sensitivity: only loans an editor marked as borrowings, plus native words. Inferred
    # labels ("derived from") can skip an intermediary, mostly toward Latin and Greek.
    expl = w.kind.isin(["explicit", "inherited"]).values
    sens = []
    for (tag, pool), pk in preds.items():
        if pool != "last":
            continue
        for name, ref in (("B+H", "B"), ("N+H", "N")):
            g = boot_gain(y[expl], pk[name][expl], base[ref][expl])
            sens.append(dict(tag=tag, model=name, vs=ref, subset="explicit labels",
                             n=int(expl.sum()), **g))
    pd.DataFrame(sens).to_csv(os.path.join(OUT, "sensitivity_explicit.csv"), index=False)
    summary["explicit_subset"] = dict(n=int(expl.sum()),
                                      classes=w[expl].label.value_counts().to_dict())

    summary["behave"] = behave_summary(w, y, classes, preds)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps(summary["disguised"], indent=1, default=float))
    for tag, m in ment.items():
        r = m["true+placebo"]
        print(tag, "n={}".format(r["n"]), {k: (round(v["coef"], 3), round(v["p"], 4))
                                         for k, v in r.items() if k != "n"})


if __name__ == "__main__":
    main()
