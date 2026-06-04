"""tool2.py — Day 2 consolidated tools.
Modules 01-13: Seq2Seq recap, attention variants, Transformer blocks, training, tokenization.
This single file replaces tool1.py through tool8.py."""

# 공용 import (consolidated)
import math
import random
from collections import Counter
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# === from tool1 (Seq2Seq recap) ===

def set_seed(s: int = 0):
    """모든 시드 고정 — 재현성 위해."""
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# 매우 작은 LSTM Seq2Seq (Day 1 복습용)
class TinyEncoder(nn.Module):
    def __init__(self, vocab, d=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.rnn = nn.LSTM(d, d, batch_first=True)

    def forward(self, x):
        # x: (B, N)  -> h_n: (1, B, d)
        e = self.emb(x)
        out, (h, c) = self.rnn(e)
        return h, c


class TinyDecoder(nn.Module):
    def __init__(self, vocab, d=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.rnn = nn.LSTM(d, d, batch_first=True)
        self.out = nn.Linear(d, vocab)

    def forward(self, y, h, c):
        # y: (B, N) -> logits: (B, N, vocab)
        e = self.emb(y)
        o, (h, c) = self.rnn(e, (h, c))
        return self.out(o), h, c


class Seq2Seq(nn.Module):
    def __init__(self, vocab, d=32):
        super().__init__()
        self.enc = TinyEncoder(vocab, d)
        self.dec = TinyDecoder(vocab, d)

    def forward(self, src, tgt):
        h, c = self.enc(src)
        out, _, _ = self.dec(tgt, h, c)
        return out


def make_copy_batch(B=32, N=8, vocab=20, pad=0, sos=1, eos=2):
    """copy task — 입력을 그대로 복사. 첫 토큰은 SOS, 마지막은 EOS."""
    body = torch.randint(3, vocab, (B, N - 2))
    src = torch.cat([torch.full((B, 1), sos), body, torch.full((B, 1), eos)], dim=1)
    tgt = src.clone()
    return src, tgt


def train_seq2seq(model, steps=400, N=8, vocab=20, lr=1e-3, device="cpu"):
    """짧은 학습 루프 — loss 기록 반환."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        src, tgt = make_copy_batch(N=N, vocab=vocab)
        src, tgt = src.to(device), tgt.to(device)
        out = model(src, tgt[:, :-1])
        loss = F.cross_entropy(out.reshape(-1, vocab), tgt[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses


def length_accuracy_curve(d=32, vocab=20, lengths=(4, 8, 16, 32), steps=200, device="cpu"):
    """길이별 도달 정확도 측정 — 길어질수록 떨어진다."""
    accs = []
    for N in lengths:
        m = Seq2Seq(vocab, d)
        train_seq2seq(m, steps=steps, N=N, vocab=vocab, device=device)
        m.eval()
        with torch.no_grad():
            src, tgt = make_copy_batch(B=64, N=N, vocab=vocab)
            src, tgt = src.to(device), tgt.to(device)
            out = m(src, tgt[:, :-1])
            pred = out.argmax(-1)
            acc = (pred == tgt[:, 1:]).float().mean().item()
        accs.append(acc)
    return list(lengths), accs


def plot_loss(losses, title="Seq2Seq loss"):
    plt.figure(figsize=(6, 3))
    plt.plot(losses); plt.title(title); plt.xlabel("step"); plt.ylabel("loss")
    plt.tight_layout(); plt.show()


def plot_length_curve(lengths, accs):
    plt.figure(figsize=(6, 3))
    plt.plot(lengths, accs, marker="o")
    plt.title("Accuracy vs Length (vanilla Seq2Seq)")
    plt.xlabel("sequence length"); plt.ylabel("accuracy")
    plt.ylim(0, 1.05); plt.grid(True); plt.tight_layout(); plt.show()


# === from tool2 (Bahdanau Additive Attention) ===

class BahdanauAttention(nn.Module):
    """e_{t,i} = v^T tanh(W1 s + W2 h)  →  alpha = softmax(e)"""

    def __init__(self, h_dec: int, h_enc: int, attn_dim: int = 64):
        super().__init__()
        self.W1 = nn.Linear(h_dec, attn_dim, bias=False)
        self.W2 = nn.Linear(h_enc, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, s, H, mask=None):
        """s: (B, h_dec), H: (B, N, h_enc), mask: (B, N) 1=keep."""
        # (B, 1, A) + (B, N, A) -> (B, N, A)
        x = self.W1(s).unsqueeze(1) + self.W2(H)
        e = self.v(torch.tanh(x)).squeeze(-1)  # (B, N)
        if mask is not None:
            e = e.masked_fill(mask == 0, -1e9)
        a = F.softmax(e, dim=-1)
        c = torch.bmm(a.unsqueeze(1), H).squeeze(1)  # (B, h_enc)
        return c, a


def show_alignment(alpha: np.ndarray, src_tokens, tgt_tokens, title="alignment"):
    """alpha: (T_tgt, N_src) numpy heatmap."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(alpha, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(src_tokens))); ax.set_xticklabels(src_tokens, rotation=45)
    ax.set_yticks(range(len(tgt_tokens))); ax.set_yticklabels(tgt_tokens)
    threshold = (alpha.max() + alpha.min()) / 2
    for i in range(alpha.shape[0]):
        for j in range(alpha.shape[1]):
            color = "white" if alpha[i, j] < threshold else "black"
            ax.text(j, i, f"{alpha[i, j]:.0%}", ha="center", va="center",
                    color=color, fontsize=9)
    ax.set_title(title); plt.tight_layout(); plt.show()


def random_alignment_demo():
    """랜덤 H, s 로 한 번 attention 굴려보고 결과 반환 — 모양 확인용."""
    torch.manual_seed(0)
    B, N, h_enc, h_dec = 2, 7, 16, 12
    H = torch.randn(B, N, h_enc)
    s = torch.randn(B, h_dec)
    attn = BahdanauAttention(h_dec, h_enc, attn_dim=8)
    c, a = attn(s, H)
    return c.shape, a.shape, a[0].detach().numpy()


# === from tool3 (Self-Attention, Q/K/V, Scaled Dot-Product — numpy + torch) ===

# ---------- numpy 버전 (직관 + 4-line 셀에서 호출) ----------
def numpy_softmax(x: np.ndarray, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def numpy_self_attention(X: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray):
    """X: (N, d) — 세 토큰 짜리 mini 예제용. attention 가중치도 함께 반환."""
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    d_k = Q.shape[-1]
    scores = Q @ K.T / math.sqrt(d_k)
    A = numpy_softmax(scores, axis=-1)
    return A @ V, A


def make_3token_demo(d=4, seed=0):
    """학생이 셀에서 호출하기 쉬운 작은 예제 입력."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((3, d))
    Wq = rng.standard_normal((d, d)) * 0.3
    Wk = rng.standard_normal((d, d)) * 0.3
    Wv = rng.standard_normal((d, d)) * 0.3
    return X, Wq, Wk, Wv


# ---------- PyTorch 버전 ----------
def scaled_dot_product_attention(Q, K, V, mask=None):
    """5-line 구현 — Module 06 의 핵심."""
    s = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
    if mask is not None:
        s = s.masked_fill(mask == 0, -1e9)
    a = s.softmax(dim=-1)
    return a @ V, a


# ---------- 시각화 ----------
def plot_attention_matrix(A, ax=None, title="attention"):
    # Accept torch/numpy input and normalize to percentage for display.
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    A = np.asarray(A, dtype=float)
    A_pct = A * 100.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(A_pct, cmap="viridis", vmin=0.0, vmax=100.0)
    ax.set_title(title); ax.set_xlabel("key"); ax.set_ylabel("query")

    for i in range(A_pct.shape[0]):
        for j in range(A_pct.shape[1]):
            ax.text(j, i, f"{A_pct[i, j]:.1f}%", ha="center", va="center",
                    color="white", fontsize=8)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label("attention (%)")
    return ax


def softmax_sharpness_demo(d_k_values=(4, 16, 64, 256)):
    """동일한 점수에 대해 d_k 가 커질수록 (스케일 없이) softmax 가 어떻게 뾰족해지는지."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal(10)
    fig, axes = plt.subplots(1, len(d_k_values), figsize=(12, 3))
    for ax, d in zip(axes, d_k_values):
        unscaled = base * math.sqrt(d / 4)
        scaled = unscaled / math.sqrt(d)
        ax.bar(range(10), numpy_softmax(unscaled), alpha=0.6, label="unscaled")
        ax.bar(range(10), numpy_softmax(scaled), alpha=0.6, label="scaled")
        ax.set_title(f"d_k={d}"); ax.set_ylim(0, 1)
    axes[0].legend(); plt.tight_layout(); plt.show()


# === from tool4 (MHA + Encoder/Decoder Block) ===

def scaled_dp_attn(Q, K, V, mask=None, dropout=None):
    s = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
    if mask is not None:
        s = s.masked_fill(mask == 0, -1e9)
    a = s.softmax(dim=-1)
    if dropout is not None:
        a = dropout(a)
    return a @ V, a


class MultiHeadAttention(nn.Module):
    """표준 MHA — d_model 을 h 등분."""

    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0
        self.h, self.d_k = h, d_model // h
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, x):
        # (B, N, d) -> (B, h, N, d_k)
        B, N, _ = x.shape
        return x.view(B, N, self.h, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        Q = self._split(self.Wq(q))
        K = self._split(self.Wk(k))
        V = self._split(self.Wv(v))
        out, _ = scaled_dp_attn(Q, K, V, mask=mask, dropout=self.drop)
        # (B, h, N, d_k) -> (B, N, d)
        B, _, N, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(B, N, self.h * self.d_k)
        return self.Wo(out)


class FeedForward(nn.Module):
    """Position-wise FFN: Linear -> GELU -> Linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class EncoderBlock(nn.Module):
    """Pre-LN 방식: x = x + sublayer(LN(x))."""

    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model); self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        a = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), mask=src_mask)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class DecoderBlock(nn.Module):
    """Masked self-attn + cross-attn + FFN."""

    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, y, enc, src_mask=None, tgt_mask=None):
        n = self.ln1(y)
        y = y + self.drop(self.self_attn(n, n, n, mask=tgt_mask))
        n = self.ln2(y)
        y = y + self.drop(self.cross_attn(n, enc, enc, mask=src_mask))
        y = y + self.drop(self.ffn(self.ln3(y)))
        return y


def causal_mask(N: int, device=None):
    """하삼각 마스크 — 디코더 self-attn 용. (1, 1, N, N).
    Note: tool4 버전을 채택 — device 인자 지원 (tool3 의 단순 버전 대체)."""
    return torch.tril(torch.ones(N, N, dtype=torch.uint8, device=device)).view(1, 1, N, N)


def padding_mask(x, pad_id: int = 0):
    """(B, N) -> (B, 1, 1, N) — broadcastable."""
    return (x != pad_id).unsqueeze(1).unsqueeze(2).to(torch.uint8)


# === from tool5 (Positional Encoding) ===

class SinusoidalPE(nn.Module):
    """Vaswani 원논문 sinusoidal PE."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, N, d) -> add (1, N, d)
        return x + self.pe[:, : x.size(1)]


class LearnedPE(nn.Module):
    """위치별 학습형 임베딩 — BERT/GPT 류."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        N = x.size(1)
        pos = torch.arange(N, device=x.device)
        return x + self.emb(pos).unsqueeze(0)


def show_sinusoidal_heatmap(d_model=64, max_len=80):
    """sin/cos heatmap — 위치 × 차원."""
    pe = SinusoidalPE(d_model, max_len).pe.squeeze(0).numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pe, aspect="auto", cmap="RdBu")
    ax.set_xlabel("dim"); ax.set_ylabel("position")
    ax.set_title("Sinusoidal Positional Encoding")
    fig.colorbar(im, ax=ax); plt.tight_layout(); plt.show()


def compare_position_distance(d_model=64, max_len=50):
    """두 PE 벡터의 코사인 유사도 행렬 — 가까운 위치는 비슷."""
    pe = SinusoidalPE(d_model, max_len).pe.squeeze(0)
    pe = pe / pe.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    sim = pe @ pe.T
    plt.figure(figsize=(5, 4))
    plt.imshow(sim.numpy(), cmap="viridis")
    plt.title("PE cosine similarity")
    plt.xlabel("position"); plt.ylabel("position")
    plt.colorbar(); plt.tight_layout(); plt.show()


# === from tool6 (Full Transformer assembly) ===

class TinyTransformer(nn.Module):
    """학습 가능한 작은 enc-dec Transformer.
    src/tgt 가 같은 vocab 을 공유하는 가장 단순한 형태."""

    def __init__(self, vocab: int, d_model: int = 64, h: int = 4,
                 d_ff: int = 128, n_layers: int = 2, dropout: float = 0.1,
                 pad_id: int = 0, max_len: int = 64):
        super().__init__()
        self.pad_id = pad_id
        self.src_emb = nn.Embedding(vocab, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(vocab, d_model, padding_idx=pad_id)
        self.pe = SinusoidalPE(d_model, max_len)
        self.encoder = nn.ModuleList([EncoderBlock(d_model, h, d_ff, dropout) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(d_model, h, d_ff, dropout) for _ in range(n_layers)])
        self.norm_e = nn.LayerNorm(d_model)
        self.norm_d = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab, bias=False)
        self.d_model = d_model

    def encode(self, src):
        m = padding_mask(src, self.pad_id)
        x = self.pe(self.src_emb(src) * math.sqrt(self.d_model))
        for blk in self.encoder:
            x = blk(x, src_mask=m)
        return self.norm_e(x), m

    def decode(self, tgt, enc, src_mask):
        N = tgt.size(1)
        tm = causal_mask(N, device=tgt.device) & padding_mask(tgt, self.pad_id)
        y = self.pe(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        for blk in self.decoder:
            y = blk(y, enc, src_mask=src_mask, tgt_mask=tm)
        return self.norm_d(y)

    def forward(self, src, tgt):
        enc, src_mask = self.encode(src)
        dec = self.decode(tgt, enc, src_mask)
        return self.proj(dec)

    @torch.no_grad()
    def greedy_decode(self, src, max_len: int = 20, sos: int = 1, eos: int = 2):
        self.eval()
        enc, src_mask = self.encode(src)
        ys = torch.full((src.size(0), 1), sos, dtype=torch.long, device=src.device)
        for _ in range(max_len - 1):
            logits = self.proj(self.decode(ys, enc, src_mask))
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)
            if (nxt == eos).all():
                break
        return ys


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# === from tool7 (Train hand-coded Transformer) ===

PAD, SOS, EOS = 0, 1, 2


def make_batch(B=64, N=10, vocab=20, task="copy"):
    """src: <SOS> tokens <EOS>, tgt: copy 또는 reverse 한 결과 (역시 SOS/EOS 포함)."""
    body = torch.randint(3, vocab, (B, N - 2))
    src = torch.cat([torch.full((B, 1), SOS), body, torch.full((B, 1), EOS)], dim=1)
    if task == "reverse":
        body = body.flip(dims=[1])
    tgt = torch.cat([torch.full((B, 1), SOS), body, torch.full((B, 1), EOS)], dim=1)
    return src, tgt


def train(model, steps=500, B=64, N=10, vocab=20, task="copy", lr=1e-3, device="cpu", log_every=20):
    """학습 루프 — loss 기록 반환. log_every 마다 print."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for s in range(steps):
        src, tgt = make_batch(B=B, N=N, vocab=vocab, task=task)
        src, tgt = src.to(device), tgt.to(device)
        out = model(src, tgt[:, :-1])
        loss = F.cross_entropy(out.reshape(-1, vocab), tgt[:, 1:].reshape(-1), ignore_index=PAD)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if (s + 1) % log_every == 0:
            print(f"step {s+1:4d}  loss={loss.item():.4f}")
    return losses


def evaluate_accuracy(model, B=64, N=10, vocab=20, task="copy", device="cpu"):
    model.eval()
    src, tgt = make_batch(B=B, N=N, vocab=vocab, task=task)
    src, tgt = src.to(device), tgt.to(device)
    pred = model.greedy_decode(src, max_len=N)
    # 잘라서 길이 맞춤
    n = min(pred.size(1), tgt.size(1))
    return (pred[:, :n] == tgt[:, :n]).float().mean().item()


def plot_losses(losses, title="train loss"):
    plt.figure(figsize=(6, 3))
    plt.plot(losses); plt.title(title)
    plt.xlabel("step"); plt.ylabel("loss")
    plt.tight_layout(); plt.show()


def quick_demo(device="cpu", task="copy"):
    """노트북에서 한 줄 호출용."""
    model = TinyTransformer(vocab=20, d_model=64, h=4, d_ff=128, n_layers=2)
    losses = train(model, steps=300, task=task, device=device)
    acc = evaluate_accuracy(model, task=task, device=device)
    return model, losses, acc


# === from tool8 (Tokenization helpers) ===

def show_tokens(tokenizer, text: str):
    """토큰 ID, 토큰 문자열을 함께 보여줌."""
    enc = tokenizer(text, return_tensors=None)
    ids = enc["input_ids"]
    toks = tokenizer.convert_ids_to_tokens(ids)
    return list(zip(ids, toks))


def compare_tokenizers(text: str, names: List[str]):
    """여러 모델의 tokenizer 결과 비교."""
    from transformers import AutoTokenizer

    rows = []
    for n in names:
        tk = AutoTokenizer.from_pretrained(n)
        toks = tk.tokenize(text)
        rows.append((n, len(toks), toks))
    return rows


# ---------- 미니 BPE (시연용) ----------
def get_pair_freq(corpus: List[List[str]]):
    """각 인접 토큰 쌍의 등장 횟수."""
    c = Counter()
    for w in corpus:
        for i in range(len(w) - 1):
            c[(w[i], w[i + 1])] += 1
    return c


def merge_pair(corpus: List[List[str]], pair):
    a, b = pair; merged = a + b
    out = []
    for w in corpus:
        new, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                new.append(merged); i += 2
            else:
                new.append(w[i]); i += 1
        out.append(new)
    return out


def train_bpe(words: List[str], num_merges: int = 10):
    """단어 리스트 → 문자열 단위로 쪼갠 뒤 BPE merge."""
    corpus = [list(w) + ["</w>"] for w in words]
    merges = []
    for _ in range(num_merges):
        freq = get_pair_freq(corpus)
        if not freq:
            break
        best = max(freq, key=freq.get)
        merges.append(best)
        corpus = merge_pair(corpus, best)
    return merges, corpus


# === from module 20 (BLEU score) ===

def _count_ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def modified_precision(hyp: List[str], refs: List[List[str]], n: int) -> float:
    """클리핑된 n-gram precision — 반복 단어 부풀리기 방지."""
    hyp_cnt = _count_ngrams(hyp, n)
    max_ref: Counter = Counter()
    for ref in refs:
        for ng, cnt in _count_ngrams(ref, n).items():
            max_ref[ng] = max(max_ref[ng], cnt)
    clipped = sum(min(cnt, max_ref[ng]) for ng, cnt in hyp_cnt.items())
    total = sum(hyp_cnt.values())
    return clipped / total if total > 0 else 0.0


def brevity_penalty(hyp_len: int, ref_len: int) -> float:
    """BP = 1 if c >= r, else exp(1 - r/c)."""
    return 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len)


def bleu(hypothesis, references, max_n: int = 4) -> float:
    """BLEU-N score. 문자열 또는 토큰 리스트 입력 모두 지원."""
    hyp = hypothesis.split() if isinstance(hypothesis, str) else list(hypothesis)
    refs = [r.split() if isinstance(r, str) else list(r) for r in references]
    log_sum = 0.0
    for n in range(1, max_n + 1):
        p = modified_precision(hyp, refs, n)
        if p == 0:
            return 0.0
        log_sum += math.log(p) / max_n
    best_ref = min(refs, key=lambda r: abs(len(r) - len(hyp)))
    bp = brevity_penalty(len(hyp), len(best_ref))
    return bp * math.exp(log_sum)


def plot_ngram_bars(hyp: List[str], ref: List[str], max_n: int = 4,
                    title: str = "N-gram Precision"):
    """n=1..max_n 에 대한 (클리핑된) precision 막대 그래프."""
    ns = list(range(1, max_n + 1))
    ps = [modified_precision(hyp, [ref], n) for n in ns]
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(ns, ps, color='#2b7cff', alpha=0.8, width=0.5)
    for bar, p in zip(bars, ps):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.02, f'{p:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xlabel('n'); ax.set_ylabel('precision')
    ax.set_ylim(0, 1.15); ax.set_xticks(ns)
    ax.set_xticklabels([f'{n}-gram' for n in ns])
    ax.set_title(title); plt.tight_layout(); plt.show()
    return ps


def bleu_comparison_table(hypotheses: List[str], reference: str) -> List[dict]:
    """여러 가설에 대해 BP, P1~P4, BLEU-4 를 한꺼번에 계산."""
    ref = reference.split()
    rows = []
    for hyp_str in hypotheses:
        hyp = hyp_str.split()
        bp = brevity_penalty(len(hyp), len(ref))
        ps = [modified_precision(hyp, [ref], n) for n in range(1, 5)]
        score = bleu(hyp, [ref])
        rows.append({
            'hypothesis': hyp_str,
            'BP': round(bp, 3),
            **{f'P{n}': round(ps[n - 1], 3) for n in range(1, 5)},
            'BLEU-4': round(score, 4),
        })
    return rows
