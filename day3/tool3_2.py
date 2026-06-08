"""
tool3_2.py — Shared visualization and display helpers for Day 3 Anderson AI Seminar.
Pure Python + matplotlib / seaborn / sklearn / PIL / numpy. No HuggingFace dependencies.

Sections:
  Part 1  (01–05): AutoTokenizer, AutoModel, Embeddings, Trainer, Datasets
  Part 2  (06–10): Text Classification, Fine-tune, NER, QA, Zero-Shot
  Part 3  (11–15): Translation, Summarization, Generation, Sentence-BERT, Table QA
  Part 4  (16–20): Image Classification, CLIP, Object Detection, Segmentation, Depth
  Part 5  (21–25): Captioning, Text-to-Image, VQA, Document QA, SAM
  Part 6  (26–31): Whisper ASR, Multimodal QA, Semantic Search, EDA, Gallery, Gemma
"""

import textwrap
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import platform
from matplotlib import rc
from matplotlib.gridspec import GridSpec


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — 01_autotokenizer · 02_automodel · 03_bert_embeddings
#           04_trainer_api · 05_datasets_hub
# ═══════════════════════════════════════════════════════════════════════════════

def get_03_bert_embedding_data():
    words = [
        # 동물 (그룹 0)
        '강아지', '고양이', '토끼', '사자', '호랑이',
        # 음식 (그룹 1)
        '김치', '비빔밥', '삼겹살', '떡볶이', '냉면',
        # 도시 (그룹 2)
        '서울', '부산', '인천', '대구', '광주',
        ]
    groups = ['동물']*5 + ['음식']*5 + ['도시']*5
    return words, groups


def _configure_korean_matplotlib_font() -> None:
    """Configure matplotlib font for Korean labels across major OSes."""
    preferred_fonts = {
        "Darwin": ["AppleGothic", "NanumGothic", "Arial Unicode MS"],
        "Windows": ["Malgun Gothic", "NanumGothic", "Arial Unicode MS"],
        "Linux": ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "UnDotum"],
    }

    os_name = platform.system()
    installed_fonts = {font.name for font in fm.fontManager.ttflist}
    candidates = preferred_fonts.get(os_name, []) + ["NanumGothic", "DejaVu Sans"]

    for font_name in candidates:
        if font_name in installed_fonts:
            plt.rcParams["font.family"] = font_name
            break

    plt.rcParams["axes.unicode_minus"] = False


def visualize_tokens(text: str, tokenizer) -> None:
    """Print each token wrapped in brackets with alternating colors (terminal-friendly)."""
    tokens = tokenizer.tokenize(text)
    colors = ["\033[44m\033[97m", "\033[42m\033[30m"]  # blue-bg / green-bg
    reset = "\033[0m"

    print(f"입력 텍스트: {text!r}")
    print(f"토큰 수: {len(tokens)}")
    print()

    parts = []
    for i, tok in enumerate(tokens):
        color = colors[i % 2]
        parts.append(f"{color}[{tok}]{reset}")
    print(" ".join(parts))
    print()

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 1.2), 1.5))
    ax.set_xlim(0, len(tokens))
    ax.set_ylim(0, 1)
    ax.axis("off")
    palette = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]
    for i, tok in enumerate(tokens):
        c = palette[i % len(palette)]
        ax.add_patch(mpatches.FancyBboxPatch(
            (i + 0.05, 0.2), 0.85, 0.6,
            boxstyle="round,pad=0.05", facecolor=c, edgecolor="white", linewidth=1.5
        ))
        ax.text(i + 0.48, 0.5, tok, ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
    ax.set_title(f"Token Visualization: {len(tokens)} Tokens", fontsize=12, pad=6)
    plt.tight_layout()
    plt.show()


def visualize_tokens(text: str, tokenizer) -> None:
    """Print each token wrapped in brackets with alternating colors (terminal-friendly)."""
    tokens = tokenizer.tokenize(text)
    colors = ["\033[44m\033[97m", "\033[42m\033[30m"]  # blue-bg / green-bg
    reset = "\033[0m"

    print(f"입력 텍스트: {text!r}")
    print(f"토큰 수: {len(tokens)}")
    print()

    parts = []
    for i, tok in enumerate(tokens):
        color = colors[i % 2]
        parts.append(f"{color}[{tok}]{reset}")
    print(" ".join(parts))
    print()

    # ─── 한글 깨짐 방지 폰트 설정 시스템 추가 ───
    os_name = platform.system()
    if os_name == "Darwin":  # macOS (현재 에러 로그를 보니 Mac을 쓰고 계시네요! 🍏)
        plt.rc("font", family="AppleGothic")
    elif os_name == "Windows":  # Windows
        plt.rc("font", family="Malgun Gothic")
    else:  # Linux / Colab / JupyterHub 등
        plt.rc("font", family="NanumGothic")

    # 마이너스 기호 깨짐 방지
    plt.rcParams["axes.unicode_minus"] = False
    # ───────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 1.2), 1.5))
    ax.set_xlim(0, len(tokens))
    ax.set_ylim(0, 1)
    ax.axis("off")
    palette = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]
    for i, tok in enumerate(tokens):
        c = palette[i % len(palette)]
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (i + 0.05, 0.2),
                0.85,
                0.6,
                boxstyle="round,pad=0.05",
                facecolor=c,
                edgecolor="white",
                linewidth=1.5,
            )
        )
        ax.text(
            i + 0.48,
            0.5,
            tok,
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold",
        )
    ax.set_title(
        f"Token Visualization: {len(tokens)} Tokens", fontsize=12, pad=6
    )
    plt.tight_layout()
    plt.show()

    
def plot_token_similarity(emb1, emb2, label1: str = "문맥 1", label2: str = "문맥 2") -> None:
    """Print and bar-chart cosine similarity between two embedding vectors."""
    e1 = np.array(emb1, dtype=float).flatten()
    e2 = np.array(emb2, dtype=float).flatten()
    cos_sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))

    print(f"코사인 유사도: {label1!r} vs {label2!r}")
    print(f"  → {cos_sim:.4f}  ({'매우 유사' if cos_sim > 0.9 else '다른 의미' if cos_sim < 0.7 else '어느 정도 유사'})")

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar([f"{label1}\nvs\n{label2}"], [cos_sim], color="#4E79A7", edgecolor="white")
    ax.set_ylim(0, 1)
    ax.axhline(0.9, color="green", linestyle="--", linewidth=1, label="매우 유사 (0.9)")
    ax.axhline(0.7, color="orange", linestyle="--", linewidth=1, label="유사 경계 (0.7)")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("같은 단어, 다른 문맥 → 다른 임베딩!")
    ax.legend(fontsize=8)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{cos_sim:.4f}", ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

def plot_token_similarity(emb1, emb2, label1: str = "문맥 1", label2: str = "문맥 2") -> None:
    """Print and bar-chart cosine similarity between two embedding vectors with Korean font support."""
    # 한글 깨짐 방지를 위한 폰트 설정
    import platform
    os_name = platform.system()
    if os_name == "Windows":
        rc("font", family="Malgun Gothic")
    elif os_name == "Darwin":  # macOS
        rc("font", family="AppleGothic")
    else:  # Linux (Colab 등)
        rc("font", family="NanumBarunGothic")
        
    # 마이너스 부호 깨짐 방지
    plt.rcParams["axes.unicode_minus"] = False

    e1 = np.array(emb1, dtype=float).flatten()
    e2 = np.array(emb2, dtype=float).flatten()
    cos_sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))

    print(f"코사인 유사도: {label1!r} vs {label2!r}")
    print(f"  → {cos_sim:.4f}  ({'매우 유사' if cos_sim > 0.9 else '다른 의미' if cos_sim < 0.7 else '어느 정도 유사'})")

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar([f"{label1}\nvs\n{label2}"], [cos_sim], color="#4E79A7", edgecolor="white")
    ax.set_ylim(0, 1)
    ax.axhline(0.9, color="green", linestyle="--", linewidth=1, label="매우 유사 (0.9)")
    ax.axhline(0.7, color="orange", linestyle="--", linewidth=1, label="유사 경계 (0.7)")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("같은 단어, 다른 문맥 → 다른 임베딩!")
    ax.legend(fontsize=8)
    
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{cos_sim:.4f}", ha="center", fontsize=12, fontweight="bold")
                
    plt.tight_layout()
    plt.show()


def print_outputs0_text_visualization(tokens, last_hidden_state, preview_dims: int = 4) -> None:
    """Print a text-only view of outputs[0] (last_hidden_state)."""
    if len(last_hidden_state.shape) != 3:
        raise ValueError("last_hidden_state must have shape (batch, seq_len, hidden_size).")
    if preview_dims <= 0:
        raise ValueError("preview_dims must be a positive integer.")

    batch_size, seq_len, hidden_size = last_hidden_state.shape
    preview_dims = min(preview_dims, hidden_size)

    if len(tokens) != seq_len:
        print(f"[warn] token count ({len(tokens)}) != seq_len ({seq_len}). Showing min length only.")

    print("=== Text visualization of outputs[0] ===")
    print("outputs[0] = last_hidden_state (batch, seq_len, hidden_size)")
    print(f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")

    max_batch_to_show = min(batch_size, 1)
    max_tokens_to_show = min(len(tokens), seq_len)
    for b in range(max_batch_to_show):
        print(f"batch[{b}]")
        for idx in range(max_tokens_to_show):
            tok = str(tokens[idx])
            preview = ", ".join(f"{v:.3f}" for v in last_hidden_state[b, idx, :preview_dims].tolist())
            print(f"  token[{idx:2d}] {tok:12s} -> hidden[{hidden_size}] [{preview}, ...]")


def visualize_model_architecture_text(model, show_full_repr: bool = True, max_depth: int = 2, max_children: int = 20) -> None:
    """Print model architecture as full text and a compact module tree."""
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0.")
    if max_children <= 0:
        raise ValueError("max_children must be > 0.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 80)
    print(f"Model class: {model.__class__.__name__}")
    if hasattr(model, "config"):
        cfg = model.config
        hidden = getattr(cfg, "hidden_size", "N/A")
        layers = getattr(cfg, "num_hidden_layers", "N/A")
        heads = getattr(cfg, "num_attention_heads", "N/A")
        print(f"Config: hidden_size={hidden}, num_hidden_layers={layers}, num_attention_heads={heads}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print("=" * 80)

    if show_full_repr:
        print("\n=== Full model architecture (repr) ===")
        print(model)

    print("\n=== Model architecture tree (compact) ===")

    def _print_tree(module, prefix: str, depth: int) -> None:
        if depth >= max_depth:
            return
        children = list(module.named_children())
        if not children:
            return

        shown_children = children[:max_children]
        for idx, (name, child) in enumerate(shown_children):
            is_last = idx == len(shown_children) - 1
            connector = "`-- " if is_last else "|-- "
            child_params = sum(p.numel() for p in child.parameters())
            print(f"{prefix}{connector}{name}: {child.__class__.__name__} ({child_params:,} params)")
            next_prefix = prefix + ("    " if is_last else "|   ")
            _print_tree(child, next_prefix, depth + 1)

        if len(children) > max_children:
            omitted = len(children) - max_children
            print(f"{prefix}`-- ... ({omitted} more child modules omitted)")

    root_name = model.__class__.__name__
    print(root_name)
    _print_tree(model, "", 0)


def plot_embeddings_2d(
    words: list,
    embeddings_2d,
    groups: list = None,
    explained_variance_ratio=None,
    title: str = "BERT 임베딩 2D 시각화 (PCA)",
) -> None:
    """Scatter-plot 2-D PCA embeddings with Korean-safe labels."""
    _configure_korean_matplotlib_font()
    emb = np.array(embeddings_2d)
    palette = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
               "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"]

    if groups is None:
        groups = [0] * len(words)

    unique_groups = sorted(set(groups))
    color_map = {g: palette[i % len(palette)] for i, g in enumerate(unique_groups)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for g in unique_groups:
        idx = [i for i, gr in enumerate(groups) if gr == g]
        ax.scatter(emb[idx, 0], emb[idx, 1],
                   color=color_map[g], s=120, zorder=3, label=str(g))

    for i, word in enumerate(words):
        ax.annotate(word, (emb[i, 0], emb[i, 1]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=11)

    ax.set_title(title, fontsize=13)
    if explained_variance_ratio is not None and len(explained_variance_ratio) >= 2:
        ax.set_xlabel(f"PCA 1 ({explained_variance_ratio[0] * 100:.1f}% var)")
        ax.set_ylabel(f"PCA 2 ({explained_variance_ratio[1] * 100:.1f}% var)")
    else:
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
    if len(unique_groups) > 1:
        ax.legend(title="그룹", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_similarity_heatmap(words: list, similarity_matrix) -> None:
    """Seaborn heatmap of a word-to-word cosine similarity matrix."""
    try:
        import seaborn as sns
        use_seaborn = True
    except ImportError:
        use_seaborn = False

    mat = np.array(similarity_matrix)
    fig, ax = plt.subplots(figsize=(max(8, len(words)), max(6, len(words) - 1)))
    if use_seaborn:
        sns.heatmap(mat, xticklabels=words, yticklabels=words,
                    annot=True, fmt=".2f", cmap="YlOrRd",
                    linewidths=0.5, linecolor="white", vmin=0, vmax=1, ax=ax)
    else:
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(words)))
        ax.set_yticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha="right")
        ax.set_yticklabels(words)
        for i in range(len(words)):
            for j in range(len(words)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax)
    ax.set_title("단어 간 코사인 유사도 히트맵", fontsize=13, pad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_training_loss(log_history: list) -> None:
    """Matplotlib line plot of training and evaluation loss from Trainer log_history."""
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry.get("step", len(train_steps)))
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", len(eval_steps)))
            eval_losses.append(entry["eval_loss"])

    fig, ax = plt.subplots(figsize=(8, 4))
    if train_losses:
        ax.plot(train_steps, train_losses, label="Train Loss", color="#4E79A7", linewidth=2)
    if eval_losses:
        ax.plot(eval_steps, eval_losses, label="Eval Loss", color="#E15759",
                linewidth=2, linestyle="--", marker="o")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    # Use English text by default to avoid missing-glyph warnings
    # when Korean fonts are unavailable in the runtime.
    ax.set_title("Train / Eval Loss Curve", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def display_dataset_stats(dataset) -> None:
    """Print count, columns, and sample rows for a HuggingFace DatasetDict or Dataset."""
    from datasets import DatasetDict

    print("=" * 55)
    print("  HuggingFace Dataset 통계")
    print("=" * 55)

    splits = dataset if isinstance(dataset, DatasetDict) else {"(dataset)": dataset}
    for split_name, split_data in splits.items():
        print(f"\n[Split: {split_name}]")
        print(f"  샘플 수   : {len(split_data):,}")
        print(f"  컬럼      : {split_data.column_names}")
        print(f"  피처 타입 : {dict(split_data.features)}")
        print(f"\n  --- 샘플 3개 ---")
        for i in range(min(3, len(split_data))):
            row = split_data[i]
            for col, val in row.items():
                display_val = str(val)[:80] + ("..." if len(str(val)) > 80 else "")
                print(f"    [{col}] {display_val}")
            print()
    print("=" * 55)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — 06_text_classification · 07_bert_finetune · 08_ner
#           09_qa_extractive · 10_zero_shot_classification
# ═══════════════════════════════════════════════════════════════════════════════


def show_ner_entities_from_model(model):
    id2label = model.config.id2label
    code_to_name = {
      "PS": "PERSON",
      "FD": "STUDY_FIELD",
      "TR": "THEORY",
      "AF": "ARTIFACTS",
      "OG": "ORGANIZATION",
      "LC": "LOCATION",
      "CV": "CIVILIZATION",
      "DT": "DATE",
      "TI": "TIME",
      "QT": "QUANTITY",
      "EV": "EVENT",
      "AM": "ANIMAL",
      "PT": "PLANT",
      "MT": "MATERIAL",
      "TM": "TERM",
    }
    entity_schema = []
    for idx, tag in sorted(id2label.items()):
      if tag == "O":
          continue
      bio, code = tag.split("-", 1)
      entity_schema.append({
          "tag": tag,
          # "position": bio,
          # "code": code,
          "name": code_to_name.get(code, "UNKNOWN"),
      })
    return entity_schema


def get_finbert_eval_data():
    """15 labeled Korean financial headlines for evaluating KR-FinBert-SC.

    Returns (texts, labels) where labels are 'positive' / 'negative' / 'neutral'.
    The 10 positive/negative examples come directly from the model card
    (snunlp/KR-FinBert-SC); the 5 neutral examples are hand-crafted.
    """
    data = [
        # positive — model card ground truth
        ("현대바이오, '폴리탁셀' 코로나19 치료 가능성에 19% 급등", "positive"),
        ("이수화학, 3분기 영업익 176억…전년比 80%↑", "positive"),
        ("GKL, 7년 만에 두 자릿수 매출성장 예상", "positive"),
        ("위지윅스튜디오, 콘텐츠 활약에 사상 첫 매출 1000억원 돌파", "positive"),
        ("삼성전자, 2년 만에 인도 스마트폰 시장 점유율 1위 '왕좌 탈환'", "positive"),
        # negative — model card ground truth
        ("영화관株 '코로나 빙하기' 언제 끝나나…CJ CGV 올 4000억 손실 날수도", "negative"),
        ("C쇼크에 멈춘 흑자비행…대한항공 1분기 영업적자 566억", "negative"),
        ("'1000억대 횡령·배임' 최신원 회장 구속…SK네트웍스 '경영 공백 방지 최선'", "negative"),
        ("부품 공급 차질에…기아차 광주공장 전면 가동 중단", "negative"),
        ("현대제철, 지난해 영업익 3,313억원···전년比 67.7% 감소", "negative"),
        # neutral — factual/descriptive headlines with no clear sentiment
        ("한국거래소, 다음달 증시 휴장일 일정 공지", "neutral"),
        ("금융감독원, 상반기 은행권 정기 검사 일정 발표", "neutral"),
        ("코스피, 외국인·기관 순매도 속 전일 대비 보합 마감", "neutral"),
        ("현대차, 3분기 실적 발표일을 10월 26일로 확정", "neutral"),
        ("삼성전자, 내년 상반기 신제품 공개 행사 일정 안내", "neutral"),
    ]
    texts  = [d[0] for d in data]
    labels = [d[1] for d in data]
    return texts, labels


def plot_sentiment_bars(texts, labels, scores):
    """Horizontal bar chart showing sentiment label and confidence for each text."""
    n = len(texts)
    colors = ["#4CAF50" if l.upper() in ("POSITIVE", "긍정") else "#F44336" for l in labels]
    short_texts = [t[:25] + "…" if len(t) > 25 else t for t in texts]

    fig, ax = plt.subplots(figsize=(9, max(3, n * 0.8)))
    bars = ax.barh(range(n), scores, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_texts, fontsize=11)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Confidence Score", fontsize=12)
    ax.set_title("Sentiment Classification Results", fontsize=14, fontweight="bold")

    for i, (bar, label, score) in enumerate(zip(bars, labels, scores)):
        ax.text(score + 0.02, i, f"{label}  {score:.2%}", va="center", fontsize=10)

    pos_patch = mpatches.Patch(color="#4CAF50", label="Positive / 긍정")
    neg_patch = mpatches.Patch(color="#F44336", label="Negative / 부정")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """Seaborn heatmap of a confusion matrix with count and percentage annotations."""
    from sklearn.metrics import confusion_matrix as sk_cm
    try:
        import seaborn as sns
        use_seaborn = True
    except ImportError:
        use_seaborn = False

    cm = sk_cm(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    annot = np.array(
        [[f"{cm[i,j]}\n({cm_norm[i,j]:.0%})" for j in range(len(labels))]
         for i in range(len(labels))]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    if use_seaborn:
        import seaborn as sns
        sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, ax=ax, vmin=0, vmax=1)
    else:
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, annot[i, j], ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


_NER_COLORS = {
    "PER":  "\033[1;33m",
    "ORG":  "\033[1;34m",
    "LOC":  "\033[1;32m",
    "DAT":  "\033[1;35m",
    "TIM":  "\033[1;36m",
    "NUM":  "\033[1;31m",
    "POH":  "\033[1;37m",
}
_RESET = "\033[0m"


def display_ner_result(text, entities):
    """Print text with named entities highlighted and show a summary table."""
    if not entities:
        print(text)
        return

    sorted_ents = sorted(entities, key=lambda e: e["start"])
    result = ""
    cursor = 0
    for ent in sorted_ents:
        start, end = ent["start"], ent["end"]
        if start < cursor:
            continue
        word = ent.get("word", text[start:end]).replace("##", "")
        label = ent.get("entity_group", ent.get("entity", "ENT")).replace("B-", "").replace("I-", "")
        color = _NER_COLORS.get(label, "\033[1;37m")
        result += text[cursor:start]
        result += f"{color}[{word}:{label}]{_RESET}"
        cursor = end
    result += text[cursor:]
    print(result)
    print()
    print("─" * 40)
    print(f"{'Entity':<20} {'Type':<10} {'Score':>6}")
    print("─" * 40)
    for ent in sorted_ents:
        word = ent.get("word", "").replace("##", "")
        label = ent.get("entity_group", ent.get("entity", "ENT")).replace("B-", "").replace("I-", "")
        score = ent.get("score", 0.0)
        print(f"{word:<20} {label:<10} {score:>6.2%}")

def display_qa_result(context, question, answer, start=None, end=None):
    """Print context with the answer span highlighted, plus question info."""
    print(f"Q: {question}")
    print("─" * 60)
    if start is not None and end is not None:
        highlighted = (context[:start]
                       + "\033[1;43m" + context[start:end] + "\033[0m"
                       + context[end:])
    else:
        highlighted = context.replace(answer, f"\033[1;43m{answer}\033[0m", 1)
    print(highlighted)
    print("─" * 60)
    print(f"A: \033[1;32m{answer}\033[0m\n")


def plot_zero_shot_results(labels, scores, title="Zero-Shot Classification"):
    """Horizontal bar chart of zero-shot scores sorted from highest to lowest."""
    paired = sorted(zip(scores, labels), reverse=True)
    scores_sorted = [p[0] for p in paired]
    labels_sorted = [p[1] for p in paired]

    cmap = plt.cm.get_cmap("RdYlGn", len(labels_sorted))
    colors = [cmap(i) for i in range(len(labels_sorted) - 1, -1, -1)]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels_sorted) * 0.7)))
    bars = ax.barh(range(len(labels_sorted)), scores_sorted, color=colors,
                   edgecolor="white", height=0.6)
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_yticklabels(labels_sorted, fontsize=12)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    for i, (bar, score) in enumerate(zip(bars, scores_sorted)):
        ax.text(score + 0.01, i, f"{score:.2%}", va="center", fontsize=10)

    plt.tight_layout()
    plt.show()


    
def qa_pipeline_09_qa_extractive(*, question, context, max_answer_len=30):
    encoded = tokenizer(
        question,
        context,
        return_tensors='pt',
        truncation='only_second',
        max_length=512,
        return_offsets_mapping=True,
    )
    offset_mapping = encoded.pop('offset_mapping')[0]
    sequence_ids = encoded.sequence_ids(0)

    with torch.no_grad():
        outputs = qa_model(**encoded)

    start_probs = torch.softmax(outputs.start_logits[0], dim=-1)
    end_probs = torch.softmax(outputs.end_logits[0], dim=-1)

    best_score = 0.0
    best_start = 0
    best_end = 0

    for start_idx, seq_id in enumerate(sequence_ids):
        if seq_id != 1:
            continue
        for end_idx in range(start_idx, min(start_idx + max_answer_len, len(sequence_ids))):
            if sequence_ids[end_idx] != 1:
                continue

            start_char = int(offset_mapping[start_idx][0])
            end_char = int(offset_mapping[end_idx][1])
            if end_char <= start_char:
                continue

            score = float(start_probs[start_idx] * end_probs[end_idx])
            if score > best_score:
                best_score = score
                best_start = start_char
                best_end = end_char

    return {
        'answer': context[best_start:best_end].strip(),
        'score': best_score,
        'start': best_start,
        'end': best_end,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — 11_translation · 12_summarization · 13_text_generation
#           14_sentence_bert · 15_table_qa
# ═══════════════════════════════════════════════════════════════════════════════

def display_translation_pairs(sources, translations, src_lang="KO", tgt_lang="EN"):
    """Print source and translated sentences side-by-side in a neat table."""
    col_w = 48
    header = f"{'[' + src_lang + '] 원문':<{col_w}}  {'[' + tgt_lang + '] 번역':<{col_w}}"
    sep = "-" * (col_w * 2 + 4)
    print(sep)
    print(header)
    print(sep)
    for src, tgt in zip(sources, translations):
        src_lines = textwrap.wrap(str(src), col_w)
        tgt_lines = textwrap.wrap(str(tgt), col_w)
        max_lines = max(len(src_lines), len(tgt_lines))
        src_lines += [""] * (max_lines - len(src_lines))
        tgt_lines += [""] * (max_lines - len(tgt_lines))
        for s, t in zip(src_lines, tgt_lines):
            print(f"{s:<{col_w}}  {t:<{col_w}}")
        print()
    print(sep)


def display_summary_comparison(original, summary):
    """Print original text and summary with word counts and compression ratio."""
    orig_words = len(original.split())
    summ_words = len(summary.split())
    ratio = round((1 - summ_words / orig_words) * 100, 1) if orig_words else 0
    sep = "=" * 60
    print(sep)
    print(f"[원문]  ({orig_words} 단어 / words)")
    print(sep)
    for line in textwrap.wrap(original, 60):
        print(line)
    print()
    print(sep)
    print(f"[요약]  ({summ_words} 단어 / words)  —  압축률 {ratio}% 감소")
    print(sep)
    for line in textwrap.wrap(summary, 60):
        print(line)
    print(sep)


def display_generated_texts(prompt, generated_list):
    """Display multiple generated text completions numbered and formatted."""
    sep = "-" * 60
    print(f"프롬프트 (Prompt): {prompt!r}")
    print(sep)
    for i, text in enumerate(generated_list, 1):
        if isinstance(text, dict):
            text = text.get("generated_text", str(text))
        print(f"[{i}] {text}\n")
    print(sep)


def plot_sentence_similarity_matrix(sentences, sim_matrix):
    """Seaborn heatmap of pairwise cosine similarity between sentences."""
    _configure_korean_matplotlib_font()
    try:
        import seaborn as sns
        use_seaborn = True
    except ImportError:
        use_seaborn = False

    labels = [s[:20] + "…" if len(s) > 20 else s for s in sentences]
    mat = np.array(sim_matrix)

    fig, ax = plt.subplots(figsize=(max(6, len(sentences)), max(5, len(sentences) - 1)))
    if use_seaborn:
        sns.heatmap(mat, xticklabels=labels, yticklabels=labels,
                    annot=True, fmt=".2f", cmap="YlOrRd",
                    vmin=0, vmax=1, ax=ax, linewidths=0.5)
    else:
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax)
    ax.set_title("문장 의미 유사도 행렬 (Sentence Semantic Similarity)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.show()


def load_table_qa_pipeline(model="google/tapas-base-finetuned-wtq"):
    """Load a TAPAS table-question-answering pipeline with a pandas 3.x compatibility patch.

    transformers' TAPAS tokenizer indexes table rows with `row[col_index]`, relying on
    pandas Series falling back to positional lookup for integer keys. pandas 3.x removed
    that fallback (raises KeyError), so we patch the two affected lookups to use `.iloc`.
    """
    import transformers.models.tapas.tokenization_tapas as tapas_tok
    from transformers import pipeline

    def _column_values_by_text(table, col_index):
        index_to_values = {}
        for row_index, row in table.iterrows():
            text = tapas_tok.normalize_for_match(row.iloc[col_index].text)
            index_to_values[row_index] = list(tapas_tok._get_numeric_values(text))
        return index_to_values

    def _column_values_by_numeric_value(self, table, col_index):
        table_numeric_values = {}
        for row_index, row in table.iterrows():
            cell = row.iloc[col_index]
            if cell.numeric_value is not None:
                table_numeric_values[row_index] = cell.numeric_value
        return table_numeric_values

    tapas_tok._get_column_values = _column_values_by_text
    tapas_tok.TapasTokenizer._get_column_values = _column_values_by_numeric_value

    return pipeline("table-question-answering", model=model)


def prepare_table_for_qa(df):
    """Cast a DataFrame to object dtype so the TAPAS tokenizer can store Cell objects in-place.

    pandas 3.x infers a pyarrow-backed `str` dtype for string columns, which rejects
    non-string values — but the TAPAS tokenizer overwrites cells with internal `Cell`
    objects while annotating numeric values.
    """
    return df.astype(object)


def display_table_qa_result(df, question, answer):
    """Print a pandas DataFrame and the question-answer pair below it."""
    sep = "=" * 60
    print(sep)
    print("  [데이터 테이블]")
    print(sep)
    print(df.to_string(index=False))
    print()
    print(sep)
    print(f"  질문 (Question) : {question}")
    print(f"  답변 (Answer)   : {answer}")
    print(sep)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — 16_image_classification · 17_zero_shot_image · 18_object_detection
#           19_image_segmentation · 20_depth_estimation
# ═══════════════════════════════════════════════════════════════════════════════

def display_classification_results(image, labels, scores):
    """Show PIL image and horizontal bar chart of top classification results side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Input Image", fontsize=13, fontweight="bold")

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(scores)))
    bars = axes[1].barh(range(len(labels)), scores, color=colors)
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels([l[:30] for l in labels], fontsize=11)
    axes[1].set_xlabel("Confidence Score", fontsize=11)
    axes[1].set_title("Top Predictions", fontsize=13, fontweight="bold")
    axes[1].set_xlim(0, 1.0)
    axes[1].invert_yaxis()

    for bar, score in zip(bars, scores):
        axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{score:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_clip_scores(labels, scores, image):
    """Show image + bar chart of CLIP zero-shot similarity scores."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Query Image", fontsize=13, fontweight="bold")

    sorted_pairs = sorted(zip(scores, labels), reverse=True)
    sorted_scores, sorted_labels = zip(*sorted_pairs)
    colors = ["#FF6B6B" if i == 0 else "#4ECDC4" for i in range(len(sorted_labels))]
    bars = axes[1].barh(range(len(sorted_labels)), sorted_scores, color=colors)
    axes[1].set_yticks(range(len(sorted_labels)))
    axes[1].set_yticklabels(sorted_labels, fontsize=12)
    axes[1].set_xlabel("CLIP Similarity Score", fontsize=11)
    axes[1].set_title("Zero-Shot Classification (CLIP)", fontsize=13, fontweight="bold")
    axes[1].invert_yaxis()

    for bar, score in zip(bars, sorted_scores):
        axes[1].text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                     f"{score:.4f}", va="center", fontsize=10)
    plt.tight_layout()
    plt.show()


def draw_detection_boxes(image, detections, threshold=0.8):
    """Draw PIL image with colored bounding boxes and labels for object detection."""
    from PIL import ImageDraw

    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    unique_labels = list({d["label"] for d in detections if d["score"] >= threshold})
    palette = [
        "#FF0000", "#00FF00", "#0000FF", "#FF8C00", "#9400D3",
        "#00CED1", "#FF1493", "#32CD32", "#FFD700", "#DC143C",
        "#00BFFF", "#FF6347", "#7FFF00", "#DA70D6", "#40E0D0",
    ]
    color_map = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}
    filtered = [d for d in detections if d["score"] >= threshold]

    for det in filtered:
        box = det["box"]
        label, score, color = det["label"], det["score"], color_map.get(det["label"], "#FFFFFF")
        x1, y1, x2, y2 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label} {score:.2f}"
        draw.rectangle([x1, y1 - 18, x1 + len(text) * 7, y1], fill=color)
        draw.text((x1 + 2, y1 - 16), text, fill="white")

    plt.figure(figsize=(10, 7))
    plt.imshow(img_draw)
    plt.axis("off")
    plt.title(f"Object Detection — {len(filtered)} objects found (threshold={threshold})",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return img_draw


def display_segmentation(image, segments):
    """Overlay each segment mask with a distinct color on the PIL image."""
    img_array = np.array(image.convert("RGB"))
    overlay = img_array.copy().astype(float)
    rng = random.Random(42)
    colors = [[rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255)]
              for _ in segments]
    label_names = []
    for seg, color in zip(segments, colors):
        mask = np.array(seg["mask"])
        label_names.append(seg["label"])
        for c in range(3):
            overlay[:, :, c][mask > 128] = overlay[:, :, c][mask > 128] * 0.4 + color[c] * 0.6
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    import matplotlib.patches as patches
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img_array)
    axes[0].axis("off")
    axes[0].set_title("Original Image", fontsize=13, fontweight="bold")
    axes[1].imshow(overlay)
    axes[1].axis("off")
    axes[1].set_title(f"Segmentation — {len(segments)} segments", fontsize=13, fontweight="bold")

    legend_patches = [patches.Patch(color=[c / 255 for c in color], label=label)
                      for label, color in zip(label_names[:12], colors[:12])]
    axes[1].legend(handles=legend_patches, loc="lower right", fontsize=8,
                   framealpha=0.8, ncol=2)
    plt.tight_layout()
    plt.show()


def display_depth_comparison(original_image, depth_map):
    """Show original image and depth map side by side with plasma colormap."""
    from PIL import Image as PILImage
    depth_array = np.array(depth_map)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(original_image)
    axes[0].axis("off")
    axes[0].set_title("Original Image", fontsize=13, fontweight="bold")

    im = axes[1].imshow(depth_array, cmap="plasma")
    axes[1].axis("off")
    axes[1].set_title("Depth Map (bright = close, dark = far)", fontsize=13, fontweight="bold")
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Relative Depth", fontsize=10)
    plt.suptitle("Monocular Depth Estimation", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 — 21_image_captioning · 22_text_to_image · 23_vqa
#           24_document_qa · 25_mask_generation
# ═══════════════════════════════════════════════════════════════════════════════

def display_image_caption(image, caption):
    """Show a PIL image above its generated caption text."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(caption, fontsize=13, wrap=True, pad=12)
    fig.tight_layout()
    plt.show()


def display_generated_image(image, prompt):
    """Display a generated PIL image with its text prompt as the figure title."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image)
    ax.axis("off")
    wrapped = "\n".join([prompt[i:i+70] for i in range(0, len(prompt), 70)])
    ax.set_title(f"Prompt: {wrapped}", fontsize=11, pad=10)
    fig.tight_layout()
    plt.show()


def display_vqa_result(image, questions, answers):
    """Show an image on the left and a Q&A panel on the right."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [1, 1]})
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Input Image", fontsize=12)

    axes[1].axis("off")
    y = 0.95
    for i, (q, a) in enumerate(zip(questions, answers)):
        axes[1].text(0.02, y, f"Q{i+1}: {q}", transform=axes[1].transAxes,
                     fontsize=10, va="top",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#ddeeff", edgecolor="#aaaaaa"))
        y -= 0.13
        axes[1].text(0.02, y, f"A: {a}", transform=axes[1].transAxes,
                     fontsize=10, va="top",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#aaaaaa"))
        y -= 0.13
    axes[1].set_title("Visual Q&A Results", fontsize=12)
    fig.tight_layout()
    plt.show()


def display_document_qa(image, questions, answers):
    """Show a document image with a Q&A extraction table below it."""
    fig = plt.figure(figsize=(10, 10))
    ax_img = fig.add_axes([0.05, 0.42, 0.9, 0.54])
    ax_img.imshow(image)
    ax_img.axis("off")
    ax_img.set_title("Document Image", fontsize=12)

    ax_tbl = fig.add_axes([0.05, 0.02, 0.9, 0.38])
    ax_tbl.axis("off")
    cell_text = [[q, a] for q, a in zip(questions, answers)]
    table = ax_tbl.table(cellText=cell_text,
                         colLabels=["Question", "Extracted Answer"],
                         cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for col in range(2):
        table[0, col].set_facecolor("#2c6fad")
        table[0, col].set_text_props(color="white", fontweight="bold")
    for row in range(1, len(cell_text) + 1):
        bg = "#f0f4ff" if row % 2 == 0 else "#ffffff"
        for col in range(2):
            table[row, col].set_facecolor(bg)
    ax_tbl.set_title("Extracted Information", fontsize=12, pad=8)
    plt.show()


def display_sam_mask(image, mask, point=None):
    """Show original image and SAM mask overlay side by side."""
    if isinstance(mask, list):
        mask = np.array(mask)
    if mask.ndim == 3:
        mask = mask[0]
    mask = mask.astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(image)
    if point is not None:
        axes[0].plot(point[0], point[1], "r*", markersize=18,
                     markeredgecolor="white", markeredgewidth=1.5, label="Click point")
        axes[0].legend(fontsize=10)
    axes[0].axis("off")
    axes[0].set_title("Original Image", fontsize=12)

    img_array = np.array(image).copy()
    overlay = np.zeros_like(img_array)
    overlay[mask] = [255, 80, 80]
    blended = (img_array * 0.6 + overlay * 0.4).astype(np.uint8)
    blended[~mask] = img_array[~mask]
    axes[1].imshow(blended)
    if point is not None:
        axes[1].plot(point[0], point[1], "r*", markersize=18,
                     markeredgecolor="white", markeredgewidth=1.5)
    axes[1].axis("off")
    axes[1].set_title("SAM Segmentation Mask", fontsize=12)

    patch = mpatches.Patch(color="#ff5050", alpha=0.6, label="Segmented region")
    axes[1].legend(handles=[patch], fontsize=10)
    fig.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6 — 26_whisper_asr · 27_image_text_to_text · 28_semantic_search
#           29_finetune_problem · 30_pipeline_gallery · 31_gemma_problem_solve
# ═══════════════════════════════════════════════════════════════════════════════

def display_asr_result(audio_info: dict, transcript: str, timestamps=None):
    """Print audio metadata, transcript, and optional word-level timestamps."""
    print("=" * 60)
    print("🎤  ASR 결과 (Automatic Speech Recognition Result)")
    print("=" * 60)

    sr = audio_info.get("sampling_rate", "N/A")
    array = audio_info.get("array", None)
    if array is not None and sr != "N/A":
        duration_str = f"{len(array) / sr:.2f}초"
    else:
        duration_str = "N/A"

    print(f"  샘플링 레이트 : {sr:,} Hz" if isinstance(sr, int) else f"  샘플링 레이트 : {sr}")
    print(f"  오디오 길이   : {duration_str}")
    print()
    print("📝  전사 결과 (Transcript):")
    for line in textwrap.fill(transcript, width=70).split("\n"):
        print(f"    {line}")
    print()

    if timestamps:
        print("⏱️  단어별 타임스탬프 (Word Timestamps):")
        print(f"  {'단어':<20} {'시작':>8}  {'끝':>8}")
        print("  " + "-" * 40)
        for item in timestamps:
            word = item.get("text", item.get("word", ""))
            ts = item.get("timestamp", [None, None])
            start_str = f"{ts[0]:.2f}s" if ts[0] is not None else "?"
            end_str   = f"{ts[1]:.2f}s" if ts[1] is not None else "?"
            print(f"  {word:<20} {start_str:>8}  {end_str:>8}")
    print("=" * 60)


def display_multimodal_qa(image, questions: list, answers: list):
    """Show PIL image on left and formatted Q&A conversation on right."""
    fig = plt.figure(figsize=(14, max(5, len(questions) * 1.8 + 2)))
    gs = GridSpec(1, 2, width_ratios=[1, 1.2], figure=fig)

    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(image)
    ax_img.axis("off")
    ax_img.set_title("Input Image", fontsize=12, fontweight="bold", pad=8)

    ax_qa = fig.add_subplot(gs[1])
    ax_qa.axis("off")
    ax_qa.set_title("Multimodal Q&A", fontsize=12, fontweight="bold", pad=8)

    y = 0.97
    step = 0.90 / max(len(questions), 1)
    for i, (q, a) in enumerate(zip(questions, answers)):
        ax_qa.text(0.02, y, textwrap.fill(f"Q{i+1}: {q}", width=55),
                   transform=ax_qa.transAxes, fontsize=9.5, va="top",
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="#ddeeff", edgecolor="#6699cc"))
        y -= step * 0.45
        ax_qa.text(0.05, y, textwrap.fill(f"A: {a}", width=55),
                   transform=ax_qa.transAxes, fontsize=9.5, va="top",
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="#eeffee", edgecolor="#66aa66"))
        y -= step * 0.55

    fig.tight_layout()
    plt.show()


def display_search_results(query: str, results: list, scores: list):
    """Print query and ranked results with ASCII progress-bar similarity scores."""
    BAR_WIDTH = 30
    max_score = max(scores) if scores else 1.0
    print("=" * 65)
    print(f"🔍  검색어 (Query): {query}")
    print("=" * 65)
    print(f"  {'순위':<5} {'유사도':>8}  {'막대':^32}  결과")
    print("  " + "-" * 62)
    for rank, (res, score) in enumerate(zip(results, scores), 1):
        filled = int(BAR_WIDTH * score / max_score)
        bar = "█" * filled + "░" * (BAR_WIDTH - filled)
        short = res if len(res) <= 38 else res[:35] + "…"
        print(f"  {rank:<5} {score:>8.4f}  [{bar}]  {short}")
    print("=" * 65)


def plot_dataset_eda(df):
    """Show class distribution bar chart and text-length histogram side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    label_col = "label" if "label" in df.columns else df.columns[-1]
    counts = df[label_col].value_counts().sort_index()
    label_names = {0: "부정 (0)", 1: "긍정 (1)"}
    labels_display = [label_names.get(int(k), str(k)) for k in counts.index]
    bars = ax.bar(labels_display, counts.values,
                  color=["#ff7070", "#70b8ff"][:len(counts)], edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.max() * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=11)
    ax.set_title("클래스 분포 (Class Distribution)", fontsize=13)
    ax.set_ylabel("샘플 수 (Count)")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    text_col = "document" if "document" in df.columns else df.columns[0]
    lengths = df[text_col].str.len()
    ax2.hist(lengths, bins=40, color="#a78bfa", edgecolor="white")
    ax2.axvline(lengths.mean(), color="#ef4444", linestyle="--",
                label=f"평균 {lengths.mean():.0f}자")
    ax2.set_title("텍스트 길이 분포 (Text Length)", fontsize=13)
    ax2.set_xlabel("문자 수 (Characters)")
    ax2.set_ylabel("빈도 (Frequency)")
    ax2.legend(fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("NSMC Dataset EDA", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.show()

    print(f"\n총 샘플: {len(df):,}  |  컬럼: {list(df.columns)}")
    print(f"텍스트 길이 — 평균: {lengths.mean():.0f}자  최대: {lengths.max()}자  최소: {lengths.min()}자")


def display_pipeline_gallery(tasks: list, models: list, results: list):
    """Print a formatted table of task → model → result for the pipeline gallery."""
    COL_TASK, COL_MODEL, COL_RESULT = 24, 38, 42
    total = COL_TASK + COL_MODEL + COL_RESULT + 6
    header = f"  {'Task':<{COL_TASK}} {'Model':<{COL_MODEL}} {'Result':<{COL_RESULT}}"
    divider = "  " + "-" * (total - 2)
    icons = ["🎭", "🌐", "❓", "🖼️", "📦", "💬", "🔍", "✍️", "🔊", "🧩"]

    print("\n" + "=" * total)
    print("  🤗  HuggingFace Pipeline Gallery")
    print("=" * total)
    print(header)
    print(divider)

    for i, (task, model, result) in enumerate(zip(tasks, models, results)):
        icon = icons[i % len(icons)]
        result_short = str(result)[:COL_RESULT - 5] + "…" if len(str(result)) > COL_RESULT - 2 else str(result)
        model_short = model[:COL_MODEL - 5] + "…" if len(model) > COL_MODEL - 2 else model
        print(f"  {icon} {task:<{COL_TASK-2}} {model_short:<{COL_MODEL}} {result_short}")

    print(divider)
    print(f"  총 {len(tasks)}개 파이프라인 시연 완료!")
    print("=" * total + "\n")


def display_gemma_response(prompt: str, response: str):
    """Print prompt in a framed box and Gemma response with clean formatting."""
    WIDTH = 70

    def _box(title, text):
        wrapped_lines = []
        for line in text.split("\n"):
            wrapped_lines.extend(textwrap.wrap(line, width=WIDTH - 4) or [""])
        print(f"\n┌─ {title} {'─' * (WIDTH - len(title) - 4)}┐")
        for wl in wrapped_lines:
            print(f"│  {wl:<{WIDTH - 4}}  │")
        print(f"└{'─' * WIDTH}┘")

    _box("📨  프롬프트 (Prompt)", prompt)
    print(f"\n{'▼' * 3}  Gemma 응답 (Response)  {'▼' * 3}")
    print("─" * WIDTH)
    for line in response.split("\n"):
        for wrapped in textwrap.wrap(line, width=WIDTH) or [""]:
            print(f"  {wrapped}")
    print("─" * WIDTH + "\n")
