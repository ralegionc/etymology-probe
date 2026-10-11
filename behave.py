"""Does the model say the right donor when asked?

A probe shows the information can be read out of the hidden states; this asks the
model. Each word goes into

    Etymology: the English word "knife" comes from

and every donor name is scored as a continuation (" Old Norse", " French", " Latin",
...). A class with two common names ("Old French", "French") gets the log of the summed
probability. Raw scores favour names the model likes in general, so each class's mean
score over all words is subtracted (calibration); both versions are reported.

Runs on a subset: every word of the nine small classes and 300 each of French, Latin
and native, the same for every model.

Writes results/behave_<model>.parquet (one row per word, one column per class).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from transformers import AutoModelForCausalLM, AutoTokenizer

import config as C

PROMPT = 'Etymology: the English word "{w}" comes from'
NAMES = {
    "French": [" French", " Old French"], "Latin": [" Latin"], "Greek": [" Greek", " Ancient Greek"],
    "Norse": [" Old Norse", " Norse"], "Dutch": [" Dutch", " Middle Dutch"], "German": [" German"],
    "Italian": [" Italian"], "Spanish": [" Spanish"], "Arabic": [" Arabic"],
    "Japanese": [" Japanese"], "Indic": [" Sanskrit", " Hindi"],
    C.NATIVE: [" Old English", " Middle English"],
}
BIG_SAMPLE = 300


def subset(words):
    big = words.label.isin(["French", "Latin", C.NATIVE])
    parts = [words[~big]] + [g.sample(min(len(g), BIG_SAMPLE), random_state=0)
                             for _, g in words[big].groupby("label")]
    return pd.concat(parts).sort_values("word").reset_index(drop=True)


@torch.inference_mode()
def score(model, revision, words, batch=64, threads=8, net=None, tok=None):
    """Log probability of every donor name after each word's prompt, by class.

    The prompt is run once per word (left-padded, explicit positions) and its key/value
    cache is repeated across the names, so each name costs only its own tokens."""
    torch.set_num_threads(threads)
    if net is None:
        tok = AutoTokenizer.from_pretrained(model, revision=revision)
        net = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=torch.float32).eval()
    forms = [(c, f) for c, fs in NAMES.items() for f in fs]
    fids = [tok(f, add_special_tokens=False).input_ids for _, f in forms]
    multi = [j for j, ids in enumerate(fids) if len(ids) > 1]
    width = max(len(fids[j]) - 1 for j in multi) if multi else 0
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    prompts = [tok(PROMPT.format(w=w), add_special_tokens=True).input_ids for w in words]
    out = np.zeros((len(words), len(forms)))
    order = np.argsort([len(p) for p in prompts])
    t0 = time.time()
    for b in range(0, len(words), batch):
        idx = order[b:b + batch]
        T = max(len(prompts[i]) for i in idx)
        ids = torch.full((len(idx), T), pad, dtype=torch.long)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, T - len(prompts[i]):] = torch.tensor(prompts[i])
            att[r, T - len(prompts[i]):] = 1
        pos = (att.cumsum(1) - 1).clamp(min=0)
        res = net(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=True)
        lp0 = torch.log_softmax(res.logits[:, -1].float(), -1)
        for j, f in enumerate(fids):
            out[idx, j] = lp0[:, f[0]].numpy()
        if multi:
            n = len(idx)
            cache = res.past_key_values
            cache.batch_repeat_interleave(len(multi))
            cont = torch.full((n * len(multi), width), pad, dtype=torch.long)
            cmask = torch.zeros((n * len(multi), width), dtype=torch.long)
            for r in range(n):
                for q, j in enumerate(multi):
                    seq = fids[j][:-1]
                    cont[r * len(multi) + q, :len(seq)] = torch.tensor(seq)
                    cmask[r * len(multi) + q, :len(seq)] = 1
            full = torch.cat([att.repeat_interleave(len(multi), 0), cmask], 1)
            start_pos = pos[:, -1:].repeat_interleave(len(multi), 0) + 1
            cpos = start_pos + torch.arange(width)[None, :]
            lp = torch.log_softmax(net(input_ids=cont, attention_mask=full, position_ids=cpos,
                                       past_key_values=cache).logits.float(), -1)
            for r, i in enumerate(idx):
                for q, j in enumerate(multi):
                    tgt = torch.tensor(fids[j][1:])
                    row = lp[r * len(multi) + q, :len(tgt)]
                    out[i, j] += row.gather(1, tgt[:, None]).sum().item()
        if (b // batch) % 10 == 0:
            print("  {:,}/{:,}  {:.0f} words/s".format(b + len(idx), len(words),
                                                     (b + len(idx)) / (time.time() - t0)), flush=True)
    cls = list(NAMES)
    return pd.DataFrame({c: logsumexp(out[:, [k for k, (cc, _) in enumerate(forms) if cc == c]], 1)
                         for c in cls})


@torch.inference_mode()
def score_slow(net, tok, words):
    """The plain version, one full sequence per word and name; the test compares them."""
    forms = [(c, f) for c, fs in NAMES.items() for f in fs]
    out = np.zeros((len(words), len(forms)))
    for i, w in enumerate(words):
        p = tok(PROMPT.format(w=w), add_special_tokens=True).input_ids
        for j, (_, f) in enumerate(forms):
            ids = p + tok(f, add_special_tokens=False).input_ids
            lp = torch.log_softmax(net(input_ids=torch.tensor([ids])).logits[0].float(), -1)
            out[i, j] = sum(lp[k - 1, ids[k]].item() for k in range(len(p), len(ids)))
    cls = list(NAMES)
    return pd.DataFrame({c: logsumexp(out[:, [k for k, (cc, _) in enumerate(forms) if cc == c]], 1)
                         for c in cls})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    words = subset(pd.read_parquet(os.path.join(C.DATA, "words.parquet")))
    t0 = time.time()
    res = score(args.model, args.revision, words.word.tolist(), args.batch, args.threads)
    res.insert(0, "word", words.word.values)
    res.insert(1, "label", words.label.values)
    name = "{}@{}".format(args.model.split("/")[-1], args.revision or "main")
    os.makedirs(C.RESULTS, exist_ok=True)
    res.to_parquet(os.path.join(C.RESULTS, "behave_{}.parquet".format(name)), index=False)
    cls = list(NAMES)
    raw = res[cls].values
    cal = raw - raw.mean(0)
    y = res.label.map({c: i for i, c in enumerate(cls)}).values
    for nm, S in (("raw", raw), ("calibrated", cal)):
        rec = pd.Series(S.argmax(1) == y).groupby(y).mean()
        print("{} {}: balanced accuracy {:.3f}".format(name, nm, rec.mean()))
    print("{:.1f} min".format((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
