"""Linear probes for donor language, the baselines they are compared with, and the
stacked models that measure what a probe adds on top of a baseline.

Every classifier gives out-of-fold class probabilities on the same five folds
(stratified by class, grouped by stem so that "knife" and "knives" never straddle a
split). Regularisation is chosen inside each training fold on a held-out fifth of it,
so no test word ever informs a choice.

Base models
  spell     character 1-5 grams (word boundaries marked) -> logistic regression
  cnn       a character CNN, the nonlinear spelling model (charcnn.py)
  spellx    the n-gram model again, trained with 20,767 extra labelled rare words
            (Zipf < 2.5) that are never test words: spelling with more experience
  freq      Zipf frequency, length in letters, length in tokens
  sem       WordNet supersenses, part of speech, gloss embedding (semantics.py)
  H[l]      hidden state at layer l -> standardise -> PCA 256 -> logistic regression

Stacks pool the base models' out-of-fold log probabilities log-linearly: one weight
per base model and one bias per class, fit on the training folds and scored on the
held-out one. With so few parameters a pool cannot overfit the small classes, so adding
a useless model costs nothing and a gain means the added model carries information the
others lack. (A free logistic regression on all the log probabilities, tried first, has
48 x 12 weights and lost accuracy when given an uninformative extra input.)
  N         spell alone, recalibrated: the character n-gram baseline
  S         spell + cnn: spelling, two ways
  B         spell + cnn + freq + sem: everything a reader could know without history
  N+H[l]    does the model beat the n-gram baseline?
  S+H[l]    does the model beat spelling?
  B+H[l]    does the model beat spelling, frequency and meaning together?

Classes are weighted equally throughout (French has 5,030 words, Arabic 65), so the
headline metrics are balanced accuracy (mean recall over classes) and balanced log loss
(mean over classes of each class's mean negative log probability).

  python probe.py baselines
  python probe.py model --tag pythia-1.4b@main [--pool last|mean]
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import charcnn
import config as C

warnings.filterwarnings("ignore", category=ConvergenceWarning)
PROBE_DIR = os.path.join(C.RESULTS, "probe")
K_FOLDS = 5
PCA_DIM = 256
EPS = 1e-6
SPELL = ["spell", "cnn", "spellx"]
BASE = SPELL + ["freq", "sem"]
STACKS = {"N": ["spell"], "S": SPELL, "SF": SPELL + ["freq"], "SS": SPELL + ["sem"], "B": BASE}


# ---------------------------------------------------------------- data

def load_words():
    w = pd.read_parquet(os.path.join(C.DATA, "words.parquet"))
    classes = sorted(w.label.unique())
    y = w.label.map({c: i for i, c in enumerate(classes)}).values
    return w, y, classes


def folds(w, y):
    """Fold index per word, fixed once and saved, so every model sees the same split."""
    path = os.path.join(C.DATA, "folds.npy")
    if os.path.exists(path):
        f = np.load(path)
        if len(f) == len(w):
            return f
    f = np.zeros(len(w), int)
    cv = StratifiedGroupKFold(K_FOLDS, shuffle=True, random_state=0)
    for k, (_, te) in enumerate(cv.split(np.zeros(len(w)), y, w.group.values)):
        f[te] = k
    np.save(path, f)
    return f


# ---------------------------------------------------------------- metrics

def bal_logloss(y, P):
    p = np.clip(P[np.arange(len(y)), y], EPS, 1)
    return float(pd.Series(-np.log(p)).groupby(y).mean().mean())


def metrics(y, P):
    pred = P.argmax(1)
    return dict(bal_acc=balanced_accuracy_score(y, pred),
                macro_f1=f1_score(y, pred, average="macro"),
                bal_logloss=bal_logloss(y, P))


# ---------------------------------------------------------------- fitting

def logreg(c):
    return LogisticRegression(C=c, class_weight="balanced", max_iter=300, tol=1e-3)


def tune_fit(make, grid, Xtr, ytr, gtr, seed=0):
    """Pick C on a grouped fifth of the training data, then refit on all of it."""
    if len(grid) > 1:
        a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(Xtr, ytr, gtr))
        scores = []
        for c in grid:
            m = make(c).fit(Xtr[a], ytr[a])
            P = full_proba(m, Xtr[b], ytr.max() + 1)
            scores.append(bal_logloss(ytr[b], P))
        c = grid[int(np.argmin(scores))]
    else:
        c = grid[0]
    return make(c).fit(Xtr, ytr), c


def full_proba(model, X, k):
    """predict_proba with a column for every class, even one missing from training."""
    P = np.full((X.shape[0], k), EPS)
    P[:, model.classes_] = model.predict_proba(X)
    return P / P.sum(1, keepdims=True)


def oof(make, grid, X, y, fold, groups):
    k = y.max() + 1
    P = np.zeros((len(y), k))
    chosen = []
    for f in range(K_FOLDS):
        tr, te = fold != f, fold == f
        m, c = tune_fit(make, grid, X[tr], y[tr], groups[tr], seed=f)
        P[te] = full_proba(m, X[te], k)
        chosen.append(c)
    return P, chosen


def fit_pool(Ls, y, w):
    """Weights a (one per model) and biases b (one per class) minimising the
    class-weighted negative log likelihood of softmax(sum_m a_m L_m + b).
    Ls is (models, words, classes) of log probabilities."""
    M, n, k = Ls.shape
    Y = np.eye(k)[y]
    W = w.sum()

    def f(theta):
        a, b = theta[:M], theta[M:]
        s = np.tensordot(a, Ls, 1) + b
        lp = s - logsumexp(s, 1, keepdims=True)
        p = np.exp(lp)
        loss = -(w * (lp * Y).sum(1)).sum() / W
        r = w[:, None] * (p - Y) / W                      # d loss / d s
        ga = np.einsum("mic,ic->m", Ls, r)
        gb = r.sum(0)
        return loss, np.r_[ga, gb]

    theta = minimize(f, np.r_[np.ones(M) / M, np.zeros(k)], jac=True, method="L-BFGS-B").x
    return theta[:M], theta[M:]


def class_weights(y):
    n = np.bincount(y)
    return (len(y) / (len(n) * n))[y]


def stack(parts, y, fold, return_weights=False):
    """Out-of-fold log-linear pool of the parts (see the module docstring)."""
    Ls = np.log(np.clip(np.array(parts), EPS, 1))
    P = np.zeros((len(y), Ls.shape[2]))
    weights = []
    for f in range(K_FOLDS):
        tr, te = fold != f, fold == f
        a, b = fit_pool(Ls[:, tr], y[tr], class_weights(y[tr]))
        P[te] = softmax(np.tensordot(a, Ls[:, te], 1) + b, axis=1)
        weights.append(a)
    return (P, np.mean(weights, 0)) if return_weights else P


# ---------------------------------------------------------------- baselines

def spelling_features(words):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, min_df=2)
    return vec.fit_transform(words)


def spelling_extra(w, y, fold, classes, grid=(0.3, 1, 3)):
    """Out-of-fold n-gram model trained on the training fold plus every rare labelled
    word whose stem is not in the test fold."""
    rare = pd.read_parquet(os.path.join(C.DATA, "words_rare.parquet"))
    rare = rare[rare.label.isin(classes)]
    yr = rare.label.map({c: i for i, c in enumerate(classes)}).values
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, min_df=2)
    X = vec.fit_transform(pd.concat([w.word, rare.word]))
    Xc, Xr = X[:len(w)], X[len(w):]
    g = w.group.values
    k = y.max() + 1
    P = np.zeros((len(y), k))
    from scipy.sparse import vstack
    for f in range(K_FOLDS):
        tr, te = fold != f, fold == f
        ok = ~rare.group.isin(set(g[te])).values
        a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=f).split(Xc[tr], y[tr], g[tr]))
        Xa = vstack([Xc[tr][a], Xr[ok]])
        ya = np.r_[y[tr][a], yr[ok]]
        scores = [bal_logloss(y[tr][b], full_proba(logreg(c).fit(Xa, ya), Xc[tr][b], k)) for c in grid]
        c = grid[int(np.argmin(scores))]
        m = logreg(c).fit(vstack([Xc[tr], Xr[ok]]), np.r_[y[tr], yr[ok]])
        P[te] = full_proba(m, Xc[te], k)
        print("  spellx fold {} C {} extra words {:,}".format(f, c, int(ok.sum())), flush=True)
    return P


def freq_features(w):
    ntok = np.load(os.path.join(C.STATES, "pythia-70m@main.npz"))["ntok"]
    return np.c_[w.zipf.values, w.word.str.len().values, ntok]


def sem_features():
    s = np.load(os.path.join(C.DATA, "semantics.npz"))
    return np.hstack([s["lex"], s["pos"], s["gloss"], s["has_wn"][:, None]])


def baselines():
    w, y, classes = load_words()
    fold = folds(w, y)
    g = w.group.values
    out, rows = {}, []
    t0 = time.time()
    # The spelling vocabulary is fit on all words: it is unsupervised (no labels), and
    # n-grams seen only in the test fold get no weight anyway.
    X = spelling_features(w.word)
    out["spell"], cs = oof(logreg, [0.1, 0.3, 1, 3], X, y, fold, g)
    print("spell C per fold:", cs, "{:.0f} s".format(time.time() - t0), flush=True)
    sc = lambda c: make_pipeline(StandardScaler(), logreg(c))
    out["freq"], _ = oof(sc, [1.0], freq_features(w), y, fold, g)
    out["sem"], cs = oof(sc, [0.0003, 0.001, 0.003, 0.01], sem_features(), y, fold, g)
    print("sem C per fold:", cs, flush=True)
    t0 = time.time()
    path = os.path.join(PROBE_DIR, "baselines.npz")
    old = np.load(path) if os.path.exists(path) else {}
    if "cnn" in old and len(old["cnn"]) == len(y):
        out["cnn"] = old["cnn"]                 # slow and seeded; recompute by deleting the file
    else:
        out["cnn"] = charcnn.oof(w.word, y, fold, g)
        print("cnn {:.0f} s".format(time.time() - t0), flush=True)
    out["spellx"] = spelling_extra(w, y, fold, classes)
    for name, parts in STACKS.items():
        out[name] = stack([out[p] for p in parts], y, fold)
    out["prior"] = np.tile(np.bincount(y) / len(y), (len(y), 1))
    for name, P in out.items():
        rows.append(dict(model=name, **metrics(y, P)))
    os.makedirs(PROBE_DIR, exist_ok=True)
    np.savez(os.path.join(PROBE_DIR, "baselines.npz"), y=y, classes=np.array(classes), **out)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PROBE_DIR, "baselines.csv"), index=False)
    print(df.round(3).to_string(index=False))


# ---------------------------------------------------------------- probes

def probe_layer(X, y, fold, g, grid):
    make = lambda c: make_pipeline(StandardScaler(), PCA(PCA_DIM, random_state=0),
                                   StandardScaler(), logreg(c))
    return oof(make, grid, X.astype(np.float32), y, fold, g)


def model(tag, pool="last", jobs=6, grid=(0.001, 0.003, 0.01, 0.03)):
    """Out-of-fold probe probabilities for every layer. The stacks with the baselines
    are built in analysis.py, so a change to the baselines does not mean re-probing."""
    w, y, classes = load_words()
    fold = folds(w, y)
    g = w.group.values
    st = np.load(os.path.join(C.STATES, tag + ".npz"))
    H = st[pool]
    assert (st["words"] == w.word.values).all()
    t0 = time.time()
    res = Parallel(n_jobs=jobs)(delayed(probe_layer)(H[:, l], y, fold, g, list(grid))
                                for l in range(H.shape[1]))
    rows = [dict(tag=tag, pool=pool, layer=l, C=float(np.median(cs)), **metrics(y, P))
            for l, (P, cs) in enumerate(res)]
    df = pd.DataFrame(rows)
    name = "{}_{}".format(tag, pool)
    df.to_csv(os.path.join(PROBE_DIR, name + ".csv"), index=False)
    np.savez_compressed(os.path.join(PROBE_DIR, name + "_oof.npz"),
                        H=np.array([P for P, _ in res], np.float32))
    print("{} ({}), {:.1f} min".format(tag, pool, (time.time() - t0) / 60))
    print(df[["layer", "C", "bal_acc", "bal_logloss"]].round(3).to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["baselines", "model"])
    ap.add_argument("--tag")
    ap.add_argument("--pool", default="last", choices=["last", "mean"])
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    if args.what == "baselines":
        baselines()
    else:
        model(args.tag, args.pool, args.jobs)


if __name__ == "__main__":
    main()
