"""How often each word's training corpus mentions it next to an etymology claim.

Uses the infini-gram API (api.infini-gram.io), which indexes the Pile (Pythia's
training data) and Dolma 1.7 (OLMo 1's) exactly. For each word:

  count        occurrences of the word
  true         the word within WINDOW tokens of a phrase naming its own donor
               ("knife" ... "from Old Norse")
  placebo      the same for a donor it does not have, picked at random per word
  generic      the same for any etymology talk ("etymology", "loanword", ...)

and once per phrase set, the phrase count itself, so co-mentions can be compared with
what chance would give.

A word with more than MAX_CLAUSE occurrences is subsampled by the engine (the
response says approx=True), so its co-mention counts are estimates.

Results are cached line by line in data/ig_cache_<index>.jsonl, so the script resumes.
Writes data/counts_<corpus>.parquet.
"""
import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

import config as C

URL = "https://api.infini-gram.io/"
MAX_CLAUSE = 500000


def clause(phrases):
    return " OR ".join(phrases)


def placebo_class(word, label, classes):
    """A donor the word does not have, fixed by the word's hash so reruns agree."""
    others = sorted(c for c in classes if c not in (label, C.NATIVE))
    h = int(hashlib.md5(word.encode()).hexdigest(), 16)
    return others[h % len(others)]


class Engine:
    def __init__(self, index):
        self.index = index
        self.path = os.path.join(C.DATA, "ig_cache_{}.jsonl".format(index))
        self.cache = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    self.cache[r["key"]] = r
        self.lock = threading.Lock()
        self.interval = 0.0
        self.next_at = 0.0

    def throttle(self):
        with self.lock:
            now = time.time()
            at = max(now, self.next_at)
            self.next_at = at + self.interval
        time.sleep(max(0.0, at - now))

    def count(self, query, cnf=False):
        key = query + ("|cnf" if cnf else "")
        if key in self.cache:
            return self.cache[key]
        payload = dict(index=self.index, query_type="count", query=query)
        if cnf:
            payload.update(max_clause_freq=MAX_CLAUSE, max_diff_tokens=C.WINDOW)
        for attempt in range(12):
            self.throttle()
            try:
                resp = requests.post(URL, json=payload, timeout=120)
                if resp.status_code == 403:
                    # the API sits behind a rate limiter that answers 403 for a while
                    raise PermissionError("rate limited")
                r = resp.json()
                if "error" in r:
                    raise RuntimeError(r["error"])
                break
            except Exception as e:
                if attempt == 11 or (isinstance(e, RuntimeError) and attempt >= 2):
                    print("giving up on", repr(query), e, flush=True)
                    return None
                wait = 300 if isinstance(e, PermissionError) else 2 ** attempt
                print("  {} ({}), waiting {} s".format(type(e).__name__, e, wait), flush=True)
                time.sleep(wait)
        rec = dict(key=key, count=r["count"], approx=bool(r.get("approx", False)),
                   tokens=r.get("tokens"))
        with self.lock:
            self.cache[key] = rec
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pythia", choices=sorted(C.INDEX))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--interval", type=float, default=1.5,
                    help="seconds between requests; the API rate-limits bursts")
    ap.add_argument("--limit", type=int, default=0, help="first N words only (testing)")
    ap.add_argument("--only", default="count,true,placebo,generic")
    ap.add_argument("--big-sample", type=int, default=600,
                    help="words per large class (French, Latin, native); the rest are all kept")
    ap.add_argument("--offline", action="store_true",
                    help="assemble the table from the cache without asking the API")
    args = ap.parse_args()
    eng = Engine(C.INDEX[args.corpus])
    eng.interval = args.interval
    kinds = args.only.split(",")

    words = pd.read_parquet(os.path.join(C.DATA, "words.parquet"))
    classes = sorted(words.label.unique())
    if args.limit:
        words = words.sample(args.limit, random_state=0)
    words["placebo_class"] = [placebo_class(w, l, classes) for w, l in zip(words.word, words.label)]

    # phrase totals (for the chance rate) and the generic clause
    sets = {c: C.PHRASES[c] for c in classes}
    sets["generic"] = C.GENERIC
    totals = {name: sum(eng.count(p)["count"] for p in ph) for name, ph in sets.items()}

    # rare classes first, then the rest in random order, so a partial run is usable
    big = words.label.isin(["French", "Latin", C.NATIVE])
    sampled = pd.concat([g.sample(min(len(g), args.big_sample), random_state=1)
                         for _, g in words[big].groupby("label")])
    words = pd.concat([words[~big].sample(frac=1, random_state=1),
                       sampled.sample(frac=1, random_state=1)])
    # in passes: count and true co-mentions for every word first, then the controls
    query = {"count": lambda w, lab, pl: (w, False),
             "true": lambda w, lab, pl: ("{} AND {}".format(w, clause(sets[lab])), True),
             "placebo": lambda w, lab, pl: ("{} AND {}".format(w, clause(sets[pl])), True),
             "generic": lambda w, lab, pl: ("{} AND {}".format(w, clause(C.GENERIC)), True)}
    passes = [[k for k in ("count", "true") if k in kinds]] + [[k] for k in ("placebo", "generic")
                                                              if k in kinds]
    jobs = []
    for ks in passes:
        for w, lab, pl in zip(words.word, words.label, words.placebo_class):
            for k in ks:
                jobs.append((w, k) + query[k](w, lab, pl))
    todo = [j for j in jobs if (j[2] + ("|cnf" if j[3] else "")) not in eng.cache]
    if args.offline:
        todo = []
    print("{:,} queries, {:,} not cached yet".format(len(jobs), len(todo)), flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(eng.count, q, cnf) for _, _, q, cnf in todo]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 500 == 0:
                rate = i / (time.time() - t0)
                print("{:,}/{:,}  {:.1f}/s  eta {:.0f} min".format(
                    i, len(todo), rate, (len(todo) - i) / rate / 60), flush=True)

    out = words[["word", "label", "placebo_class"]].copy()
    key = {(w, k): q + ("|cnf" if cnf else "") for w, k, q, cnf in jobs}
    for kind in kinds:
        recs = [eng.cache.get(key[(w, kind)]) for w in out.word]
        out[kind] = [r["count"] if r else None for r in recs]
        if kind != "count":
            out[kind + "_approx"] = [r["approx"] if r else None for r in recs]
        else:
            out["n_llama_tokens"] = [len(r["tokens"]) if r and r.get("tokens") else None
                                     for r in recs]
    out["total_true"] = out.label.map(totals)
    out["total_placebo"] = out.placebo_class.map(totals)
    out["total_generic"] = totals["generic"]
    out = out.sort_values("word")
    if not args.limit:
        out.to_parquet(os.path.join(C.DATA, "counts_{}.parquet".format(args.corpus)), index=False)
    print(json.dumps(totals, indent=1))
    print(out.describe().to_string())


if __name__ == "__main__":
    main()
