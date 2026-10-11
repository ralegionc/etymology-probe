"""Hidden states for every word, at every layer, from one model (and one checkpoint).

Each word is read in the same neutral template, "Here is a word: knife". The prefix is
there so the word is not the first token (the first position of a causal model acts as
an attention sink and looks alike for every input). Two summaries of the word's tokens
are kept per layer: the last token (in a causal model the only one that has seen the
whole word) and the mean.

Writes data/states/<model>@<revision>.npz with float16 arrays last and mean, each
(words, layers + 1, width), plus the number of tokens each word took.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

import config as C


def tag(model, revision):
    return "{}@{}".format(model.split("/")[-1], revision or "main")


def encode(tok, words, template=C.TEMPLATE):
    """Token ids for each word in the template, and where the word's tokens sit."""
    prefix, suffix = template.split("{w}")
    assert suffix == "", "the word must end the template"
    head = tok(prefix.rstrip(" "), add_special_tokens=True).input_ids
    seqs, spans = [], []
    for w in words:
        ids = tok(" " + w, add_special_tokens=False).input_ids
        seqs.append(head + ids)
        spans.append((len(head), len(head) + len(ids)))
    return seqs, spans


@torch.inference_mode()
def run(model, revision, words, batch=64, threads=8):
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    net = AutoModel.from_pretrained(model, revision=revision, dtype=torch.float32)
    net.eval()
    seqs, spans = encode(tok, words)
    n, L = len(words), net.config.num_hidden_layers + 1
    d = net.config.hidden_size
    last = np.zeros((n, L, d), np.float16)
    mean = np.zeros((n, L, d), np.float16)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    order = np.argsort([len(s) for s in seqs])     # similar lengths together, less padding
    t0 = time.time()
    for b in range(0, n, batch):
        idx = order[b:b + batch]
        width = max(len(seqs[i]) for i in idx)
        ids = torch.full((len(idx), width), pad, dtype=torch.long)
        att = torch.zeros((len(idx), width), dtype=torch.long)
        for j, i in enumerate(idx):
            ids[j, :len(seqs[i])] = torch.tensor(seqs[i])
            att[j, :len(seqs[i])] = 1
        out = net(input_ids=ids, attention_mask=att, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 1)     # (batch, L, T, d)
        for j, i in enumerate(idx):
            a, z = spans[i]
            last[i] = hs[j, :, z - 1].numpy()
            mean[i] = hs[j, :, a:z].mean(1).numpy()
        if b == 0 or (b // batch) % 40 == 0:
            done = b + len(idx)
            print("  {:,}/{:,}  {:.0f} words/s".format(done, n, done / (time.time() - t0)),
                  flush=True)
    ntok = np.array([z - a for a, z in spans], np.int16)
    return last, mean, ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None, help="e.g. step8000 for a Pythia checkpoint")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-mean", action="store_true", help="keep only last-token states (saves disk)")
    args = ap.parse_args()
    os.makedirs(C.STATES, exist_ok=True)
    path = os.path.join(C.STATES, tag(args.model, args.revision) + ".npz")
    if os.path.exists(path) and not args.force:
        print("exists:", path)
        return
    words = pd.read_parquet(os.path.join(C.DATA, "words.parquet")).word.tolist()
    t0 = time.time()
    last, mean, ntok = run(args.model, args.revision, words, args.batch, args.threads)
    arrays = dict(last=last, ntok=ntok, words=np.array(words))
    if not args.no_mean:
        arrays["mean"] = mean
    np.savez(path, **arrays)
    print("{}: {} words, {} layers, width {}, {:.1f} min".format(
        tag(args.model, args.revision), *last.shape, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
