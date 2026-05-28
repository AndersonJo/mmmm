"""
Day 1 helper module — PyTorch 기초/RNN/LSTM/GRU/Seq2Seq 까지.
모든 노트북에서 `from tool1 import *` 로 사용.
"""
from __future__ import annotations
import os
import math
import time
import random
import string
from collections import Counter
from dataclasses import dataclass

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset

# ---------------------------------------------------------------------------
# 0. 공통 유틸
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 1. 시각화 헬퍼 (English only labels)
# ---------------------------------------------------------------------------

def plot_loss(history, title="loss"):
    fig, ax = plt.subplots(figsize=(6, 4))
    if isinstance(history, dict):
        for k, v in history.items():
            ax.plot(v, label=k)
        ax.legend()
    else:
        ax.plot(history, label="loss")
        ax.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_curves(curves: dict, xlabel="x", ylabel="y", title=""):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (xs, ys) in curves.items():
        ax.plot(xs, ys, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_image_grid(images, titles=None, cols=8, cmap="gray"):
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.4))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i < n:
            img = images[i]
            if hasattr(img, "detach"):
                img = img.detach().cpu().numpy()
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))
                if img.shape[2] == 1:
                    img = img[:, :, 0]
            ax.imshow(img, cmap=cmap)
            if titles is not None:
                ax.set_title(str(titles[i]), fontsize=8)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2. Toy regression (모듈 4, 6, 7 등에서 재사용)
# ---------------------------------------------------------------------------

def make_toy_regression(n=200, noise=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.linspace(-3, 3, n).unsqueeze(1)
    y = 0.7 * x + 0.4 * torch.sin(2.0 * x) + noise * torch.randn(n, 1, generator=g)
    return x, y


def make_classification_blobs(n_per=120, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]])
    xs, ys = [], []
    for i, c in enumerate(centers):
        xs.append(c + 0.7 * torch.randn(n_per, 2, generator=g))
        ys.append(torch.full((n_per,), i, dtype=torch.long))
    x = torch.cat(xs); y = torch.cat(ys)
    perm = torch.randperm(len(x), generator=g)
    return x[perm], y[perm]


# ---------------------------------------------------------------------------
# 3. 간단 학습 루프 (모듈 7~10)
# ---------------------------------------------------------------------------

def train_simple(model, x, y, loss_fn, optimizer, epochs=100, verbose=False):
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"epoch {epoch:4d} | loss {loss.item():.4f}")
    return losses


def train_classifier(model, loader, val_loader, optimizer, loss_fn, epochs=5, device="cpu"):
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        running = 0.0; n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb); n += len(xb)
        train_loss = running / n
        val_loss, val_acc = evaluate_classifier(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"epoch {epoch+1}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.3f}")
    return history


@torch.no_grad()
def evaluate_classifier(model, loader, loss_fn, device="cpu"):
    model.eval()
    running = 0.0; correct = 0; n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = loss_fn(out, yb)
        running += loss.item() * len(xb)
        pred = out.argmax(dim=1)
        correct += (pred == yb).sum().item()
        n += len(xb)
    return running / n, correct / n


# ---------------------------------------------------------------------------
# 4. 작은 모델들
# ---------------------------------------------------------------------------

class TinyMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=32, out_dim=3, dropout=0.0, use_bn=False):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden)]
        if use_bn: layers.append(nn.BatchNorm1d(hidden))
        layers += [nn.ReLU()]
        if dropout > 0: layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden, hidden)]
        if use_bn: layers.append(nn.BatchNorm1d(hidden))
        layers += [nn.ReLU()]
        if dropout > 0: layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 5. 시계열/문자 데이터 (모듈 11~14)
# ---------------------------------------------------------------------------

def make_sine_sequence(seq_len=50, n=400, freqs=(0.6, 1.1), seed=0):
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for _ in range(n):
        f = freqs[0] + (freqs[1] - freqs[0]) * torch.rand(1, generator=g).item()
        phase = 2 * math.pi * torch.rand(1, generator=g).item()
        t = torch.linspace(0, 4 * math.pi, seq_len + 1)
        s = torch.sin(f * t + phase)
        xs.append(s[:-1].unsqueeze(-1))
        ys.append(s[1:].unsqueeze(-1))
    return torch.stack(xs), torch.stack(ys)


TINY_SHAKESPEARE = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles,\n"
    "And by opposing end them. To die, to sleep,\n"
    "No more; and by a sleep to say we end\n"
    "The heart-ache, and the thousand natural shocks\n"
    "That flesh is heir to: 'tis a consummation\n"
    "Devoutly to be wish'd. To die, to sleep,\n"
    "To sleep, perchance to dream — ay, there's the rub.\n"
) * 6


class CharDataset(Dataset):
    def __init__(self, text: str, seq_len: int = 32):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# Seq2Seq toy data: reverse a number string
def make_reverse_pairs(n=2000, max_len=6, seed=0):
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        L = rng.randint(3, max_len)
        s = "".join(rng.choice(string.digits) for _ in range(L))
        pairs.append((s, s[::-1]))
    return pairs


class Seq2SeqVocab:
    PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"

    def __init__(self, chars: str = string.digits):
        toks = [self.PAD, self.BOS, self.EOS] + list(chars)
        self.stoi = {t: i for i, t in enumerate(toks)}
        self.itos = {i: t for t, i in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def encode(self, s, add_bos=False, add_eos=False):
        ids = []
        if add_bos: ids.append(self.stoi[self.BOS])
        ids += [self.stoi[c] for c in s]
        if add_eos: ids.append(self.stoi[self.EOS])
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            t = self.itos[int(i)]
            if t in (self.PAD, self.BOS): continue
            if t == self.EOS: break
            out.append(t)
        return "".join(out)


def collate_seq2seq(batch, vocab: Seq2SeqVocab):
    src = [torch.tensor(vocab.encode(s), dtype=torch.long) for s, _ in batch]
    tgt = [torch.tensor(vocab.encode(t, add_bos=True, add_eos=True), dtype=torch.long) for _, t in batch]
    src_len = torch.tensor([len(s) for s in src])
    tgt_len = torch.tensor([len(t) for t in tgt])
    src_pad = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=vocab.stoi[vocab.PAD])
    tgt_pad = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=vocab.stoi[vocab.PAD])
    return src_pad, src_len, tgt_pad, tgt_len


# ---------------------------------------------------------------------------
# 6. Quick MNIST/FashionMNIST helper (lazy import torchvision)
# ---------------------------------------------------------------------------

def get_mnist_loaders(batch_size=128, root="./data", fashion=False, num_workers=0):
    from torchvision import datasets, transforms
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    cls = datasets.FashionMNIST if fashion else datasets.MNIST
    train = cls(root=root, train=True, download=True, transform=tfm)
    test = cls(root=root, train=False, download=True, transform=tfm)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


__all__ = [
    "os", "math", "time", "random", "string", "Counter", "dataclass",
    "np", "plt", "matplotlib",
    "torch", "nn", "F", "Dataset", "DataLoader", "random_split", "TensorDataset",
    "set_seed", "get_device", "count_params", "trainable_params",
    "plot_loss", "plot_curves", "show_image_grid",
    "make_toy_regression", "make_classification_blobs",
    "train_simple", "train_classifier", "evaluate_classifier",
    "TinyMLP",
    "make_sine_sequence", "TINY_SHAKESPEARE", "CharDataset",
    "make_reverse_pairs", "Seq2SeqVocab", "collate_seq2seq",
    "get_mnist_loaders",
]
