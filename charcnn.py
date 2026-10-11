"""A second spelling model: a small character CNN.

The n-gram logistic regression is linear in its features, and an LM has seen far more
spelling than these 11,000 words. If the LM's advantage were only better spelling, a
nonlinear model of the letters should take part of it back. This one reads "^knife$"
as characters, with filters 2 to 5 letters wide, max-pooled, trained with class-weighted
cross-entropy and stopped early on a grouped fifth of the training fold. Three seeds are
averaged.
"""
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn

MAXLEN = 24


def encode(words):
    chars = sorted({c for w in words for c in w})
    idx = {c: i + 3 for i, c in enumerate(chars)}     # 0 pad, 1 start, 2 end
    X = np.zeros((len(words), MAXLEN), np.int64)
    for i, w in enumerate(words):
        s = [1] + [idx[c] for c in w[:MAXLEN - 2]] + [2]
        X[i, :len(s)] = s
    return X, len(chars) + 3


class Net(nn.Module):
    def __init__(self, vocab, k, dim=32, filters=96, widths=(2, 3, 4, 5)):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(dim, filters, w, padding=w - 1) for w in widths)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(filters * len(widths), k)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        mask = (x == 0).unsqueeze(1)
        hs = []
        for conv in self.convs:
            h = torch.relu(conv(e))[:, :, :x.shape[1]]
            hs.append(h.masked_fill(mask, -1e4).max(2).values)
        return self.out(self.drop(torch.cat(hs, 1)))


def fit_predict(Xtr, ytr, gtr, Xte, k, vocab, seed, epochs=40, patience=6):
    torch.manual_seed(seed)
    a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(Xtr, ytr, gtr))
    n = np.bincount(ytr[a], minlength=k)
    cw = torch.tensor(len(a) / (k * np.maximum(n, 1)), dtype=torch.float32)
    net = Net(vocab, k)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    xa, ya = torch.tensor(Xtr[a]), torch.tensor(ytr[a])
    xb, yb = torch.tensor(Xtr[b]), torch.tensor(ytr[b])
    best, best_state, bad = np.inf, None, 0
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        net.train()
        for i in np.array_split(rng.permutation(len(a)), max(1, len(a) // 128)):
            opt.zero_grad()
            loss_fn(net(xa[i]), ya[i]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(xb), yb).item()
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {kk: t.clone() for kk, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(torch.tensor(Xte)), 1).numpy()


def oof(words, y, fold, groups, seeds=(0, 1, 2), threads=4):
    torch.set_num_threads(threads)
    X, vocab = encode(list(words))
    k = y.max() + 1
    P = np.zeros((len(y), k))
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        P[te] = np.mean([fit_predict(X[tr], y[tr], groups[tr], X[te], k, vocab, s)
                         for s in seeds], 0)
    return P
