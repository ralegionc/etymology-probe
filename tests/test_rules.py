"""The rules the README describes, as worked examples. No downloaded data or models
needed; the labelling tests need the sibling loanword-flow-map code (not its data)."""
import numpy as np
import pytest
from scipy.special import softmax

import analysis as A
import build_words as BW
import config as C
import corpus_counts as CC
import probe as P

L, BE = C.flowmap()


# ---------------------------------------------------------------- labels

def hops(*rows):
    return [(r, lang, term) for r, lang, term in rows]


def test_norse_loan_behind_middle_english():
    # "From Middle English thei, from Old Norse þeir": the loan is the first hop out
    h = hops(("inherited_from", "Middle English", "þei"), ("derived_from", "Old Norse", "þeir"))
    ev = BE.walk("English", "they", h)
    assert ev is not None and L.lineage(ev["donor"]) == "Old Norse"
    assert C.CLASS_OF[L.lineage(ev["donor"])] == "Norse"


def test_native_word_reaches_old_english():
    h = hops(("inherited_from", "Middle English", "water"), ("inherited_from", "Old English", "wæter"),
             ("inherited_from", "Proto-West Germanic", "*watar"))
    assert BE.walk("English", "water", h) is None
    assert BW.native(h, L)


def test_word_of_unknown_origin_is_not_native():
    # "boy": Middle English boie, of uncertain origin. No Old English, no label.
    h = hops(("inherited_from", "Middle English", "boie"),)
    assert BE.walk("English", "boy", h) is None
    assert not BW.native(h, L)


def test_inherited_word_with_a_later_norse_hop_is_a_loan():
    # "take": the Old English line is listed first, then Old Norse taka. The walk takes
    # the first hop out of English, as the flow map does.
    h = hops(("inherited_from", "Middle English", "taken"), ("inherited_from", "Old English", "tacan"),
             ("derived_from", "Old Norse", "taka"))
    ev = BE.walk("English", "take", h)
    assert ev is not None and ev["donor"] == "Old Norse"


def test_dutch_and_low_german_are_one_class():
    assert C.CLASS_OF[L.lineage("Middle Dutch")] == C.CLASS_OF[L.lineage("Middle Low German")] == "Dutch"
    assert C.CLASS_OF[L.lineage("Anglo-Norman")] == "French"


def test_word_pattern():
    assert BW.WORD.match("knife") and not BW.WORD.match("Knife") and not BW.WORD.match("x")
    assert not BW.WORD.match("café") and not BW.WORD.match("t-shirt")


# ---------------------------------------------------------------- corpus counts

def test_placebo_is_stable_and_never_the_true_class():
    classes = ["Arabic", "Dutch", "French", "Norse", C.NATIVE]
    for word, label in [("knife", "Norse"), ("boss", "Dutch"), ("water", C.NATIVE)]:
        p = CC.placebo_class(word, label, classes)
        assert p == CC.placebo_class(word, label, classes)
        assert p not in (label, C.NATIVE)


def test_phrase_clauses_fit_the_api():
    # infini-gram accepts at most four terms per OR clause
    assert all(len(v) <= 4 for v in C.PHRASES.values()) and len(C.GENERIC) <= 4


# ---------------------------------------------------------------- pooling

def make_probs(rng, y, k, signal):
    logits = rng.normal(size=(len(y), k))
    logits[np.arange(len(y)), y] += signal
    return softmax(logits, axis=1)


def test_pool_ignores_noise_and_uses_signal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 3000)
    good = make_probs(rng, y, 4, 2.0)
    noise = softmax(rng.normal(size=(3000, 4)), axis=1)
    Ls = np.log(np.array([good, noise]))
    a, b = P.fit_pool(Ls, y, np.ones(len(y)))
    assert a[0] > 0.5 and abs(a[1]) < 0.1


def test_adding_noise_to_a_stack_costs_almost_nothing():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 4000)
    fold = rng.integers(0, P.K_FOLDS, 4000)
    good = make_probs(rng, y, 5, 1.5)
    noise = softmax(rng.normal(size=(4000, 5)), axis=1)
    one = P.bal_logloss(y, P.stack([good], y, fold))
    two = P.bal_logloss(y, P.stack([good, noise], y, fold))
    assert two < one + 0.01


def test_complementary_models_pool_better_than_either():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, 4000)
    fold = rng.integers(0, P.K_FOLDS, 4000)
    a = make_probs(rng, y, 3, 1.0)
    b = make_probs(rng, y, 3, 1.0)       # independent noise: together they know more
    both = P.bal_logloss(y, P.stack([a, b], y, fold))
    assert both < min(P.bal_logloss(y, P.stack([a], y, fold)),
                      P.bal_logloss(y, P.stack([b], y, fold))) - 0.05


def test_layer_selection_uses_training_folds_only():
    fold = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    stacked = np.zeros((3, 10, 2))
    for l in range(3):
        stacked[l, :, 0] = l          # mark which layer a prediction came from
    train_loss = np.ones((P.K_FOLDS, 3))
    train_loss[:, 2] = 0              # layer 2 best on every fold's training data...
    train_loss[4, 1] = -1             # ...except fold 4's, where layer 1 is
    Q, chosen = A.select(stacked, train_loss, fold)
    assert chosen == [2, 2, 2, 2, 1]
    assert (Q[fold == 4, 0] == 1).all() and (Q[fold == 0, 0] == 2).all()


def test_bootstrap_gain_sign():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 2000)
    better = make_probs(rng, y, 3, 2.0)
    worse = make_probs(rng, y, 3, 0.5)
    g = A.boot_gain(y, better, worse, draws=200)
    assert g["acc_gain"] > 0 and g["ll_gain"] > 0 and g["ll_lo"] > 0


def test_balanced_logloss_weights_classes_equally():
    y = np.array([0] * 90 + [1] * 10)
    Q = np.full((100, 2), 0.5)
    Q[:90] = [0.9, 0.1]               # good on the big class only
    # mean over classes of -log p: (-log .9 + -log .5) / 2
    assert P.bal_logloss(y, Q) == pytest.approx((-np.log(0.9) - np.log(0.5)) / 2)
