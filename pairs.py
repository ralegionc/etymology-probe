"""Dedicated two-class tests for the loans that should look native: Norse vs native and
Dutch vs native.

The twelve-class probe has to separate everything at once; a contrast as fine as
"they" (Norse) vs "the" (native) may not be one of the directions it keeps. Here every
model is retrained on the two classes only, on the same folds restricted to those words:

  spell   character n-grams, logistic regression
  base    spell + the twelve-class baselines' log odds for the pair (cnn, spellx, freq, sem)
  H[l]    hidden state at layer l -> PCA 128 -> logistic regression
  base+H  logistic regression on the base and H[l] out-of-fold log odds

The layer is chosen inside each training fold, as in analysis.py. Scores are AUCs with
bootstrap intervals; positive class is the loan.

Writes results/pairs_binary.csv.
"""
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import analysis as A
import config as C
import probe as P

PAIRS = A.PAIRS
DIM = 128


def lr(c):
    return LogisticRegression(C=c, class_weight="balanced", max_iter=500)


def oof_binary(make, grid, X, t, fold, g):
    p = np.zeros(len(t))
    for f in range(P.K_FOLDS):
        tr, te = fold != f, fold == f
        m, _ = P.tune_fit(make, grid, X[tr], t[tr], g[tr], seed=f)
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def logit(p):
    p = np.clip(p, P.EPS, 1 - P.EPS)
    return np.log(p) - np.log1p(-p)


def pooled(cols, t, fold):
    X = np.column_stack(cols)
    p = np.zeros(len(t))
    for f in range(P.K_FOLDS):
        tr, te = fold != f, fold == f
        m = lr(1.0).fit(X[tr], t[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def delta_ci(t, s1, s0, draws=1000, seed=0):
    """Paired bootstrap of AUC(s1) - AUC(s0) over the same resampled words."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
    d = []
    for _ in range(draws):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        d.append(roc_auc_score(t[i], s1[i]) - roc_auc_score(t[i], s0[i]))
    return roc_auc_score(t, s1) - roc_auc_score(t, s0), np.percentile(d, 2.5), np.percentile(d, 97.5)


def auc_ci(t, s, draws=400, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
    b = [roc_auc_score(t[i], s[i]) for i in
         (np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(draws))]
    return roc_auc_score(t, s), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main(tags=None):
    w, y, classes = P.load_words()
    fold = P.folds(w, y)
    base = np.load(os.path.join(P.PROBE_DIR, "baselines.npz"))
    spell_X = P.spelling_features(w.word)
    if tags is None:
        tags = [t for t in A.tags("last") if t.endswith("@main")
                and os.path.exists(os.path.join(C.STATES, t + ".npz"))]
    rows = []
    for a, b in PAIRS:
        ia, ib = classes.index(a), classes.index(b)
        m = (y == ia) | (y == ib)
        t = (y[m] == ia).astype(int)
        fm, gm = fold[m], w.group.values[m]
        s_spell = oof_binary(lr, [0.3, 1, 3], spell_X[m], t, fm, gm)
        odds = [np.log(np.clip(base[k][m, ia], P.EPS, 1)) - np.log(np.clip(base[k][m, ib], P.EPS, 1))
                for k in ("cnn", "spellx", "freq", "sem")]
        s_base = pooled([logit(s_spell)] + odds, t, fm)
        for name, s in (("spell", s_spell), ("base", s_base)):
            rows.append(dict(pair="{} vs {}".format(a, b), tag="-", model=name,
                             **dict(zip(["auc", "lo", "hi"], auc_ci(t, s)))))
        for tag in tags:
            H = np.load(os.path.join(C.STATES, tag + ".npz"))["last"][m]
            make = lambda c: make_pipeline(StandardScaler(), PCA(DIM, random_state=0),
                                           StandardScaler(), lr(c))
            per_layer = [oof_binary(make, [0.003, 0.01, 0.03], H[:, l].astype(np.float32), t, fm, gm)
                         for l in range(H.shape[1])]
            # choose the layer per fold on the training folds' AUC
            sH, sBH = np.zeros(len(t)), np.zeros(len(t))
            BH = [pooled([logit(s_base), logit(p)], t, fm) for p in per_layer]
            lh, lbh = [], []
            for f in range(P.K_FOLDS):
                tr, te = fm != f, fm == f
                lh.append(int(np.argmax([roc_auc_score(t[tr], p[tr]) for p in per_layer])))
                sH[te] = per_layer[lh[-1]][te]
                lbh.append(int(np.argmax([roc_auc_score(t[tr], p[tr]) for p in BH])))
                sBH[te] = BH[lbh[-1]][te]
            for name, s, ref, layers in (("H", sH, s_spell, lh), ("base+H", sBH, s_base, lbh)):
                d = delta_ci(t, s, ref)
                rows.append(dict(pair="{} vs {}".format(a, b), tag=tag, model=name, layers=str(layers),
                                 **dict(zip(["auc", "lo", "hi"], auc_ci(t, s))),
                                 vs="spell" if name == "H" else "base",
                                 **dict(zip(["delta", "delta_lo", "delta_hi"], d))))
            print(a, b, tag, {r["model"]: (round(r["auc"], 3), round(r["delta"], 3),
                                          round(r["delta_lo"], 3)) for r in rows[-2:]}, flush=True)
        print(a, b, "spell {:.3f} base {:.3f}".format(roc_auc_score(t, s_spell), roc_auc_score(t, s_base)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.RESULTS, "pairs_binary.csv"), index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
