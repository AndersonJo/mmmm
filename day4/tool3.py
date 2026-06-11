"""tool3.py — Day 3 consolidated tools.
HuggingFace day: pipelines, Korean models, classification, NER, QA, summarization,
translation, generation, embeddings, similarity, RAG, ASR, datasets, PEFT/LoRA.

Korean BERT/GPT2/koBART/koElectra/koSRoBERTa 등 작은 모델 위주.
"""

import math
import os
import platform
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_KOREAN_FONT_CONFIGURED = False


def _configure_korean_matplotlib_font() -> None:
    """Configure matplotlib font for Korean labels across major OSes."""
    global _KOREAN_FONT_CONFIGURED
    if _KOREAN_FONT_CONFIGURED:
        return

    preferred_fonts = {
        "Darwin": ["AppleGothic", "NanumGothic", "Arial Unicode MS"],
        "Windows": ["Malgun Gothic", "NanumGothic", "Arial Unicode MS"],
        "Linux": ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "UnDotum"],
    }

    os_name = platform.system()
    installed_fonts = {font.name for font in fm.fontManager.ttflist}
    candidates = preferred_fonts.get(os_name, []) + ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]
    available = [font_name for font_name in candidates if font_name in installed_fonts]

    if available:
        existing_sans = plt.rcParams.get("font.sans-serif", [])
        merged_sans = available + [f for f in existing_sans if f not in available]
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = merged_sans
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"

    plt.rcParams["axes.unicode_minus"] = False
    _KOREAN_FONT_CONFIGURED = True


# === 공용 ===

def set_seed(s: int = 0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# === 토크나이저/모델 로딩 (HF) ===

def load_tokenizer(name: str):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    if "kogpt2" in name.lower():
        # AutoTokenizer picks a byte-level GPT2 tokenizer here that mis-encodes
        # Korean text (round-trip produces mojibake). The model card's own
        # tokenizer needs explicit special tokens to encode/decode correctly.
        return PreTrainedTokenizerFast.from_pretrained(
            name, bos_token='</s>', eos_token='</s>', unk_token='<unk>',
            pad_token='<pad>', mask_token='<mask>'
        )
    return AutoTokenizer.from_pretrained(name)


def load_model(name: str, task: str = "auto"):
    """task: 'auto' | 'mlm' | 'cls' | 'token' | 'qa' | 's2s' | 'lm' """
    from transformers import (
        AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification,
        AutoModelForTokenClassification, AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM, AutoModelForCausalLM
    )
    table = {
        "auto": AutoModel,
        "mlm": AutoModelForMaskedLM,
        "cls": AutoModelForSequenceClassification,
        "token": AutoModelForTokenClassification,
        "qa": AutoModelForQuestionAnswering,
        "s2s": AutoModelForSeq2SeqLM,
        "lm": AutoModelForCausalLM,
    }
    return table[task].from_pretrained(name)


def model_size_mb(model) -> float:
    n = sum(p.numel() for p in model.parameters())
    return n * 4 / 1024 / 1024  # float32 가정


# === 시각화 helpers ===

def show_topk_bar(labels: List[str], scores: List[float], title="top-k"):
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(range(len(labels))[::-1], scores, color="#2b7cff")
    ax.set_yticks(range(len(labels))[::-1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("probability"); ax.set_title(title)
    ax.set_xlim(0, max(1.0, max(scores) * 1.1))
    plt.tight_layout(); plt.show()


def show_logits_softmax(logits: np.ndarray, labels=None, title="logits -> softmax"):
    _configure_korean_matplotlib_font()
    p = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    x = range(len(logits))
    axs[0].bar(x, logits, color="#ff7043"); axs[0].set_title("raw logits")
    axs[1].bar(x, p, color="#2b7cff"); axs[1].set_title("softmax")
    if labels is not None:
        for ax in axs: ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    plt.tight_layout(); plt.show()


def show_attention_mask_grid(mask: np.ndarray, title="mask"):
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mask, cmap="gray_r")
    ax.set_title(title); ax.set_xlabel("key"); ax.set_ylabel("query")
    plt.tight_layout(); plt.show()


def show_2d_scatter(X: np.ndarray, labels=None, texts=None, title="2D"):
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(6, 5))
    if labels is None:
        ax.scatter(X[:, 0], X[:, 1], s=40, c="#2b7cff")
    else:
        for c in sorted(set(labels)):
            m = np.array(labels) == c
            ax.scatter(X[m, 0], X[m, 1], s=40, label=str(c))
        ax.legend(fontsize=8)
    if texts is not None:
        for i, t in enumerate(texts):
            ax.annotate(t, (X[i, 0], X[i, 1]), fontsize=8)
    ax.set_title(title); ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    plt.tight_layout(); plt.show()


# === MLM (fill-mask) ===

@torch.no_grad()
def predict_mask_topk(model, tokenizer, sentence: str, k: int = 5):
    """문장 내 [MASK] 위치의 top-k 후보 반환."""
    inputs = tokenizer(sentence, return_tensors="pt")
    mask_id = tokenizer.mask_token_id

    # [mask] 위치
    pos = (inputs.input_ids == mask_id).nonzero(as_tuple=True)[1]
    out = model(**inputs).logits[0, pos[0]]
    probs = F.softmax(out, dim=-1)
    topk = probs.topk(k)
    toks = tokenizer.convert_ids_to_tokens(topk.indices.tolist())
    return list(zip(toks, topk.values.tolist()))


# === 분류 (classification) ===

@torch.no_grad()
def classify_text(model, tokenizer, text: str, label_map=None):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    logits = model(**inputs).logits[0]
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    idx = int(probs.argmax())
    label = label_map[idx] if label_map else str(idx)
    return label, float(probs[idx]), probs


# === NER ===

@torch.no_grad()
def predict_ner(model, tokenizer, text: str, id2label=None):
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = inputs.pop("offset_mapping")[0].tolist()
    out = model(**inputs).logits[0]
    pred = out.argmax(-1).tolist()
    toks = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    items = []
    for tk, p, off in zip(toks, pred, offsets):
        lab = id2label[p] if id2label else str(p)
        items.append((tk, lab, off))
    return items


# === Extractive QA ===

@torch.no_grad()
def answer_question(model, tokenizer, question: str, context: str):
    inputs = tokenizer(question, context, return_tensors="pt")
    out = model(**inputs)
    start_probs = F.softmax(out.start_logits[0], dim=-1)
    end_probs = F.softmax(out.end_logits[0], dim=-1)
    s = int(start_probs.argmax())
    e = int(end_probs.argmax())
    answer = tokenizer.decode(inputs["input_ids"][0][s:e + 1], skip_special_tokens=True)
    score = float(start_probs[s] * end_probs[e])
    return {"answer": answer, "score": score, "start": s, "end": e}


# === Embeddings & similarity ===

@torch.no_grad()
def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    s = (last_hidden * mask).sum(dim=1)
    n = mask.sum(dim=1).clamp_min(1e-9)
    return s / n


@torch.no_grad()
def encode_texts(model, tokenizer, texts: List[str], pool: str = "mean", normalize: bool = True):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    out = model(**inputs)
    h = out.last_hidden_state
    if pool == "mean":
        emb = mean_pool(h, inputs.attention_mask)
    else:
        emb = h[:, 0]
    if normalize:
        emb = F.normalize(emb, dim=-1)
    return emb.numpy()


def cosine_matrix(A: np.ndarray, B: np.ndarray = None):
    if B is None: B = A
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return An @ Bn.T


def show_sim_heatmap(sim: np.ndarray, labels: List[str], title="cosine sim"):
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(sim, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center", fontsize=9, color="white")
    ax.set_title(title)
    plt.colorbar(im, ax=ax); plt.tight_layout(); plt.show()


# === PCA (직접 구현 — 외부 의존 없이) ===

def pca_2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


# === 미니 RAG ===

class TinyVectorIndex:
    def __init__(self, embeddings: np.ndarray, docs: List[str]):
        self.E = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        self.docs = docs

    def search(self, q: np.ndarray, k: int = 3):
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self.E @ q
        top = np.argsort(-sims)[:k]
        return [(self.docs[i], float(sims[i])) for i in top]


# === Triplet loss & 미니 metric learning ===

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_ap = (anchor - positive).pow(2).sum(-1)
        d_an = (anchor - negative).pow(2).sum(-1)
        return F.relu(d_ap - d_an + self.margin).mean()


class TinyEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# === 빔 서치 (간단 구현, 시각화용) ===

@torch.no_grad()
def greedy_generate(model, tokenizer, prompt: str, max_new: int = 30, device: str = "cpu"):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
    
    # ids.shape[1]은 입력 프롬프트의 토큰 개수입니다.
    input_len = ids.shape[1]
    generated_ids = out[0][input_len:]
    
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


@torch.no_grad()
def sample_generate(model, tokenizer, prompt: str, max_new: int = 30,
                    top_k: int = 50, top_p: float = 0.9, temperature: float = 1.0,
                    device: str = "cpu"):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(
        ids, max_new_tokens=max_new, do_sample=True,
        top_k=top_k, top_p=top_p, temperature=temperature,
        pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id or 0,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# === ROUGE 미니 ===

def rouge_l(reference: str, prediction: str) -> float:
    """문자 단위 LCS 기반 ROUGE-L F1 (토큰화 없이 단순 비교용)."""
    r = reference.split(); p = prediction.split()
    if not r or not p: return 0.0
    n, m = len(r), len(p)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            dp[i+1][j+1] = dp[i][j] + 1 if r[i] == p[j] else max(dp[i+1][j], dp[i][j+1])
    lcs = dp[n][m]
    prec = lcs / m; rec = lcs / n
    if prec + rec == 0: return 0.0
    return 2 * prec * rec / (prec + rec)


# === BLEU 미니 (1-gram precision) ===

def bleu_1(reference: str, prediction: str) -> float:
    r = reference.split(); p = prediction.split()
    if not p: return 0.0
    rset = {}
    for w in r: rset[w] = rset.get(w, 0) + 1
    matches = 0; used = dict(rset)
    for w in p:
        if used.get(w, 0) > 0:
            matches += 1; used[w] -= 1
    return matches / len(p)


# === Triplet 합성 데이터 ===

def make_triplet_synth(n_classes: int = 4, per_class: int = 12, in_dim: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 2.0, (n_classes, in_dim))
    X, y = [], []
    for c in range(n_classes):
        X.append(centers[c] + rng.normal(0, 0.6, (per_class, in_dim))); y.extend([c] * per_class)
    return np.vstack(X).astype(np.float32), np.array(y)


def make_triplet_batch(X, y, B: int = 32, seed=None):
    rng = np.random.default_rng(seed)
    a_idx = rng.integers(0, len(X), B)
    p_idx = []; n_idx = []
    for ai in a_idx:
        same = np.where(y == y[ai])[0]; same = same[same != ai]
        diff = np.where(y != y[ai])[0]
        p_idx.append(rng.choice(same)); n_idx.append(rng.choice(diff))
    return X[a_idx], X[p_idx], X[n_idx]


# === BIO 태깅 ===

BIO_COLORS = {"B": "#2b7cff", "I": "#7c4dff", "O": "#9aa0a6"}


def render_bio(tokens, tags):
    """간단한 시각용 print."""
    out = []
    for t, g in zip(tokens, tags):
        out.append(f"{t}/{g}")
    return " ".join(out)


# === 작은 멜 스펙트로그램 (whisper 시연용) ===

def fake_audio(duration: float = 1.0, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 660 * t)


def plot_mel(audio: np.ndarray, sr: int = 16000):
    _configure_korean_matplotlib_font()
    n_fft = 512
    spec = np.abs(np.fft.rfft(audio[:n_fft]))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(spec); ax.set_title("naive spectrum (preview)")
    ax.set_xlabel("freq bin"); ax.set_ylabel("magnitude")
    plt.tight_layout(); plt.show()


# === LoRA 미니 ===

class LoRALinear(nn.Module):
    """학습 가능한 W (frozen) + A·B (trainable) — 본 노트북에서 직접 사용 가능."""

    def __init__(self, in_dim: int, out_dim: int, r: int = 4, alpha: float = 8):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        for p in self.W.parameters(): p.requires_grad = False
        self.A = nn.Parameter(torch.randn(in_dim, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, out_dim))
        self.scale = alpha / r

    def forward(self, x):
        return self.W(x) + (x @ self.A @ self.B) * self.scale


def count_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# === HF datasets 헬퍼 ===

def load_hf_dataset(name: str, *args, **kwargs):
    from datasets import load_dataset
    return load_dataset(name, *args, **kwargs)


# === 작은 한국어 더미 코퍼스 ===

DUMMY_KOR_DOCS = [
    "트랜스포머는 자연어 처리에서 널리 사용되는 모델 구조입니다.",
    "BERT 는 양방향 인코더이고 GPT 는 단방향 디코더입니다.",
    "한국어 형태소 분석은 어절 단위가 아니라 의미 단위로 쪼갭니다.",
    "임베딩은 단어를 벡터 공간에 배치하는 작업입니다.",
    "코사인 유사도는 두 벡터 사이 각도를 측정합니다.",
    "RAG 는 검색을 결합한 생성 방식으로 환각을 줄입니다.",
    "Whisper 는 OpenAI 가 공개한 다국어 음성 인식 모델입니다.",
    "LoRA 는 적은 파라미터로 미세조정하는 기법입니다.",
]


# === 시퀀스 reverse 토이 데이터 (from-scratch encoder-decoder transformer용) ===
# 정수 0..NUM_VOCAB-1 을 그대로 토큰 id 로 쓰고, 특수 토큰은 그 뒤에 둔다.

NUM_VOCAB = 100
PAD_ID = NUM_VOCAB        # 100
BOS_ID = NUM_VOCAB + 1    # 101
EOS_ID = NUM_VOCAB + 2    # 102
VOCAB_SIZE = NUM_VOCAB + 3


def make_reverse_batch(batch_size: int = 32, min_len: int = 3, max_len: int = 8,
                       num_vocab: int = NUM_VOCAB, seed=None):
    """무작위 정수 시퀀스와 그 역순 쌍을 만든다 (정수 하나 = 토큰 하나).

    반환 (모두 LongTensor [batch, T]):
      encoder_input_ids / encoder_attention_mask — `[seq..., EOS, PAD...]`
      decoder_input_ids                          — `[BOS, rev..., PAD...]` (teacher forcing)
      labels                                      — `[rev..., EOS, -100...]` (-100 = loss에서 무시)
    """
    rng = np.random.default_rng(seed)
    lengths = rng.integers(min_len, max_len + 1, size=batch_size)
    seqs = [rng.integers(0, num_vocab, size=L).tolist() for L in lengths]
    T = int(max(lengths)) + 1  # +1 = EOS

    enc_ids = np.full((batch_size, T), PAD_ID, dtype=np.int64)
    enc_mask = np.zeros((batch_size, T), dtype=np.int64)
    dec_in = np.full((batch_size, T), PAD_ID, dtype=np.int64)
    labels = np.full((batch_size, T), -100, dtype=np.int64)

    for i, seq in enumerate(seqs):
        rev = seq[::-1]
        enc_ids[i, :len(seq)] = seq
        enc_ids[i, len(seq)] = EOS_ID
        enc_mask[i, :len(seq) + 1] = 1

        dec_in[i, 0] = BOS_ID
        dec_in[i, 1:len(rev) + 1] = rev
        labels[i, :len(rev)] = rev
        labels[i, len(rev)] = EOS_ID

    to_t = lambda a: torch.tensor(a, dtype=torch.long)
    return to_t(enc_ids), to_t(enc_mask), to_t(dec_in), to_t(labels)


def decode_reverse_tokens(ids) -> List[int]:
    """토큰 id 시퀀스에서 EOS 이전까지의 정수만 뽑아낸다 (PAD/BOS/EOS 제거)."""
    out = []
    for tid in (ids.tolist() if torch.is_tensor(ids) else list(ids)):
        if tid == EOS_ID:
            break
        if tid not in (PAD_ID, BOS_ID):
            out.append(int(tid))
    return out


def plot_loss_curve(losses: List[float], title="training loss"):
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(losses, color="#2b7cff")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title(title)
    fig.tight_layout()
    return fig
