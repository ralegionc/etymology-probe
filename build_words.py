"""The labelled word list: every common English word whose Wiktionary etymology names a
donor (a loan) or traces it back through Old English (a native word).

The labels come from the loanword-flow-map walk (build_events.walk), run here on English
entries only. Its events table cannot be used directly: it keeps one event per
(receiving stage, word, donor), so when Middle English "þei" has its own page, the
English "they" event is dropped as a duplicate of it. That removes they, knife and
sugar, among many others.

A word is native when the walk reaches Old English (or a Proto-Germanic form by
inheritance) without leaving the English line. A word that stops in Middle English
("boy", of uncertain origin) has no label.

Writes data/words.parquet.
"""
import json
import os
import re

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from nltk.stem.snowball import SnowballStemmer
from wordfreq import zipf_frequency

import config as C

WORD = re.compile(r"^[a-z]{2,}$")


def english_hops(BE, L):
    path = os.path.join(C.LOANWORD_DIR, "data", "etymology.parquet")
    t = pq.read_table(path, columns=["term_id", "lang", "term", "reltype",
                                     "related_lang", "related_term"])
    t = t.append_column("rid", [np.arange(t.num_rows)])
    con = duckdb.connect()
    con.register("t", t)
    q = """
        with e as (select * from t where trim(lang) = 'English'),
             s as (select term_id, min(rid) stop_rid from e
                   where reltype in {stop} group by term_id)
        select e.term_id, e.rid, e.term, e.reltype, trim(e.related_lang) related_lang,
               e.related_term
        from e left join s using (term_id)
        where e.reltype in {hop} and (s.stop_rid is null or e.rid < s.stop_rid)
        order by e.term_id, e.rid
    """.format(stop=BE.STOP, hop=BE.HOP)
    hops = con.sql(q).df()
    names = hops.related_lang.dropna().unique()
    hops["related_lang"] = hops.related_lang.map(dict(zip(names, map(L.normalize, names))))
    return hops


def native(hops, L):
    """True when the hops stay in the English line and reach Old English, or go to a
    proto-language by inheritance. Call only after walk() found no loan."""
    reached_oe = False
    for reltype, rlang, _ in hops:
        if rlang is None or rlang in L.CONSTRUCTED:
            continue
        if L.is_proto(rlang):
            return reached_oe or reltype == "inherited_from"
        if L.lineage(rlang) == "English":
            reached_oe |= rlang == "Old English"
            continue
        return False
    return reached_oe


def label_entries(hops, BE, L):
    rows = []
    for tid, g in hops.groupby("term_id", sort=False):
        term = g.term.iat[0]
        if not isinstance(term, str) or not WORD.match(term):
            continue
        h = list(zip(g.reltype, g.related_lang, g.related_term))
        ev = BE.walk("English", term, h)
        if ev is not None:
            rows.append(dict(term_id=tid, word=term, donor=ev["donor"],
                             donor_lin=L.lineage(ev["donor"]), donor_term=ev["donor_term"],
                             kind=ev["kind"], reltype=ev["reltype"], recv=ev["recv"]))
        elif native(h, L):
            rows.append(dict(term_id=tid, word=term, donor=None, donor_lin=C.NATIVE,
                             donor_term=None, kind="inherited", reltype="inherited_from",
                             recv="Old English"))
    return pd.DataFrame(rows)


def build():
    L, BE = C.flowmap()
    hops = english_hops(BE, L)
    ent = label_entries(hops, BE, L)
    n = {"English entries walked": hops.term_id.nunique(), "labelled entries": len(ent)}

    ent = ent[ent.reltype != "orthographic_borrowing_from"]
    ent = ent[(ent.donor_lin == C.NATIVE) | ~ent.donor.map(L.excluded)]
    ent["label"] = ent.donor_lin.map(lambda d: C.NATIVE if d == C.NATIVE else C.CLASS_OF.get(d))
    n["entries with a donor class"] = int(ent.label.notna().sum())
    ent = ent[ent.label.notna()]

    # one spelling, one label: homographs with different origins (bank, the river side
    # and the money house) are ambiguous to anything that reads the form
    labels = ent.groupby("word").label.nunique()
    n["spellings with conflicting labels (dropped)"] = int((labels > 1).sum())
    ent = ent[ent.word.isin(labels.index[labels == 1])]
    words = ent.drop_duplicates("word").copy()

    words["zipf"] = [zipf_frequency(w, "en") for w in words.word]
    rare = words[words.zipf < C.MIN_ZIPF]
    words = words[words.zipf >= C.MIN_ZIPF]
    n["common enough (zipf >= {})".format(C.MIN_ZIPF)] = len(words)

    size = words.label.value_counts()
    keep = size.index[size >= C.MIN_CLASS]
    n["dropped classes"] = ", ".join("{} ({})".format(k, v) for k, v in size.items() if k not in keep)
    words = words[words.label.isin(keep)]
    # rare words: extra training data for the spelling baseline, never test words
    rare = rare[rare.label.isin(keep)].copy()

    # cross-validation groups: inflected and derived forms of one stem stay together
    stem = SnowballStemmer("english").stem
    words["group"] = words.word.map(stem)
    words = words.sort_values("word").reset_index(drop=True)
    os.makedirs(C.DATA, exist_ok=True)
    words.drop(columns=["term_id"]).to_parquet(os.path.join(C.DATA, "words.parquet"), index=False)
    rare["group"] = rare.word.map(stem)
    rare.drop(columns=["term_id"]).sort_values("word").to_parquet(
        os.path.join(C.DATA, "words_rare.parquet"), index=False)
    n["rare words (spelling training only)"] = len(rare)

    n["words"] = len(words)
    n["classes"] = words.label.value_counts().to_dict()
    n["kind"] = words.kind.value_counts().to_dict()
    os.makedirs(C.RESULTS, exist_ok=True)
    with open(os.path.join(C.RESULTS, "words_build.json"), "w") as f:
        json.dump(n, f, indent=1, default=int)
    for k, v in n.items():
        print("{}: {}".format(k, v))
    print("{:,} words in {} classes".format(len(words), words.label.nunique()))
    print(words.label.value_counts().to_string())
    print(pd.crosstab(words.label, words.kind).to_string())
    return words


if __name__ == "__main__":
    build()
