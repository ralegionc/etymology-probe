"""What each word means, without its spelling: the semantic control.

A probe that beats spelling could be reading topic rather than history: Japanese loans
name Japanese things, Arabic loans cluster in religion, Norse loans are everyday words.
This control gives a baseline classifier the meaning and nothing else.

For each word, from WordNet (via nltk):
  lexnames   the share of its senses in each of WordNet's 45 supersenses
             (noun.food, verb.motion, ...)
  pos        the share of its senses by part of speech
  gloss      a sentence embedding (all-MiniLM-L6-v2) of its first three definitions,
             averaged. The definitions never contain the word, so the encoder sees the
             meaning, not the form.
  has_wn     whether WordNet knows the word at all (function words and many recent
             loans are missing)

Writes data/semantics.npz.
"""
import os

import numpy as np
import pandas as pd
import torch
from nltk.corpus import wordnet as wn
from transformers import AutoModel, AutoTokenizer

import config as C

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
SENSES = 3


@torch.inference_mode()
def embed(texts, batch=128):
    tok = AutoTokenizer.from_pretrained(ENCODER)
    net = AutoModel.from_pretrained(ENCODER).eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=96,
                  return_tensors="pt")
        h = net(**enc).last_hidden_state
        m = enc.attention_mask.unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1)
        out.append(torch.nn.functional.normalize(v, dim=-1).numpy())
    return np.concatenate(out)


def features(words):
    lexnames = sorted({s.lexname() for s in wn.all_synsets()})
    poses = ["n", "v", "a", "s", "r"]
    lex = np.zeros((len(words), len(lexnames)), np.float32)
    pos = np.zeros((len(words), len(poses)), np.float32)
    has = np.zeros(len(words), np.float32)
    glosses = {}
    for i, w in enumerate(words):
        syn = wn.synsets(w)
        if not syn:
            continue
        has[i] = 1
        for s in syn:
            lex[i, lexnames.index(s.lexname())] += 1 / len(syn)
            pos[i, poses.index(s.pos())] += 1 / len(syn)
        glosses[i] = [s.definition() for s in syn[:SENSES]]
    flat = [(i, g) for i, gs in glosses.items() for g in gs]
    vec = embed([g for _, g in flat])
    gloss = np.zeros((len(words), vec.shape[1]), np.float32)
    n = np.zeros(len(words))
    for (i, _), v in zip(flat, vec):
        gloss[i] += v
        n[i] += 1
    gloss[n > 0] /= n[n > 0, None]
    return dict(lex=lex, pos=pos, gloss=gloss, has_wn=has, lexnames=np.array(lexnames))


def main():
    words = pd.read_parquet(os.path.join(C.DATA, "words.parquet"))
    f = features(words.word.tolist())
    np.savez(os.path.join(C.DATA, "semantics.npz"), words=words.word.values, **f)
    print("{:,} words, {:.0%} in WordNet".format(len(words), f["has_wn"].mean()))
    print(pd.Series(f["has_wn"], index=words.label.values).groupby(level=0).mean().round(2).to_string())


if __name__ == "__main__":
    main()
