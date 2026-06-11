"""tool4.py — Day 4 consolidated tools.
Part A (01-08): Manufacturing time-series + Transformer (CMAPSS RUL).
Part B (09-20): Vision Transformers (ViT, DINO, Swin, CLIP).
All notebooks: `from tool4 import *`."""

import io
import math
import os
import random
import time
import urllib.request
import zipfile
import copy
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ---------- 공용 ----------

def set_seed(s: int = 0):
    """시드 고정."""
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def pick_device():
    """cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------
# Part A: CMAPSS / 시계열 RUL
# ---------------------------------------------------------------

CMAPSS_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
DATA_DIR = Path.home() / ".cache" / "cmapss"
SENSOR_NAMES = [f"s{i}" for i in range(1, 22)]
SETTINGS_NAMES = ["op1", "op2", "op3"]
COLS = ["unit", "cycle"] + SETTINGS_NAMES + SENSOR_NAMES


def synthesize_cmapss(n_engines: int = 80, max_cycle: int = 220,
                      n_sensors: int = 21, seed: int = 0) -> np.ndarray:
    """합성 CMAPSS 데이터 생성. NASA 다운로드 실패시 fallback.
    각 엔진은 다른 길이를 갖고, 시간이 갈수록 센서값이 drift + noise.
    Returns: shape (rows, 26) — [unit, cycle, op1..3, s1..21]"""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_engines + 1):
        L = int(rng.integers(120, max_cycle))
        t = np.arange(L)
        # 각 sensor: baseline + drift + cycle + noise
        sensors = []
        for k in range(n_sensors):
            base = rng.normal(0, 1)
            drift = (t / L) ** (1 + 0.5 * rng.random()) * rng.normal(0, 1) * 2
            cyc = 0.3 * np.sin(2 * np.pi * t / (15 + 5 * rng.random()))
            noise = rng.normal(0, 0.15, size=L)
            sensors.append(base + drift + cyc + noise)
        sensors = np.stack(sensors, axis=1)  # (L, n_sensors)
        ops = rng.normal(0, 0.05, size=(L, 3))
        unit_col = np.full((L, 1), u)
        cycle_col = (np.arange(L) + 1).reshape(-1, 1)
        block = np.concatenate([unit_col, cycle_col, ops, sensors], axis=1)
        rows.append(block)
    return np.concatenate(rows, axis=0)


from pathlib import Path
import urllib.request
import zipfile
from typing import Optional
import numpy as np


def download_cmapss(force: bool = False) -> Optional[Path]:
    """CMAPSS 다운로드 시도. 실패하면 None."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_fd001 = DATA_DIR / "train_FD001.txt"

    if train_fd001.exists() and not force:
        return train_fd001

    try:
        zip_path = DATA_DIR / "cmapss.zip"
        urllib.request.urlretrieve(CMAPSS_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)
        if train_fd001.exists():
            return train_fd001
    except Exception as e:
        print(f"[tool4] CMAPSS 다운로드 실패: {e}")

    return None


def load_cmapss() -> pd.DataFrame:
    """학습용 CMAPSS 데이터 로드 (DataFrame). 파일이 없거나 다운로드 실패 시 RuntimeError 발생."""
    path = download_cmapss()
    if path is not None:
        return pd.DataFrame(np.loadtxt(path), columns=COLS)

    raise RuntimeError(
        "CMAPSS 데이터를 로드할 수 없습니다. 파일 다운로드 상태를 확인해 주세요."
    )


def _to_array(data) -> np.ndarray:
    """DataFrame이면 ndarray로, 아니면 그대로 반환 (load_cmapss/synthesize_cmapss 겸용)."""
    if isinstance(data, pd.DataFrame):
        return data.to_numpy(dtype=np.float64)
    return data


def make_rul_labels(arr: np.ndarray, clip: int = 125) -> np.ndarray:
    """piecewise-linear RUL: max_cycle - cycle, 그리고 clip 이상은 잘라냄."""
    arr = _to_array(arr)
    rul = np.zeros(arr.shape[0])
    for u in np.unique(arr[:, 0]):
        mask = arr[:, 0] == u
        cycles = arr[mask, 1]
        max_c = cycles.max()
        rul[mask] = np.clip(max_c - cycles, 0, clip)
    return rul


def make_windows(arr: np.ndarray, rul: np.ndarray, win: int = 30,
                 stride: int = 1, sensor_cols: Optional[List[int]] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """각 엔진별 sliding window. Returns (X: B,T,F  ;  y: B)."""
    if sensor_cols is None:
        sensor_cols = list(range(5, arr.shape[1]))  # skip unit, cycle, ops
    X, y = [], []
    for u in np.unique(arr[:, 0]):
        idx = np.where(arr[:, 0] == u)[0]
        block = arr[idx][:, sensor_cols]
        rblock = rul[idx]
        for s in range(0, len(idx) - win + 1, stride):
            X.append(block[s:s + win])
            y.append(rblock[s + win - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------- 탐색적 분석 (EDA) ----------

def describe_sensors(arr: np.ndarray):
    """센서별 기초 통계량 (mean/std/min/max/skew/kurtosis/변동계수)."""
    from scipy import stats as sstats
    arr = _to_array(arr)
    sensor_data = arr[:, 5:5 + len(SENSOR_NAMES)]
    rows = []
    for i, name in enumerate(SENSOR_NAMES):
        col = sensor_data[:, i]
        rows.append({
            "sensor": name,
            "mean": col.mean(), "std": col.std(),
            "min": col.min(), "max": col.max(),
            "skew": sstats.skew(col), "kurtosis": sstats.kurtosis(col),
            "cv": col.std() / (abs(col.mean()) + 1e-8),
        })
    return pd.DataFrame(rows)


def engine_life_summary(arr: np.ndarray) -> np.ndarray:
    """엔진별 수명(cycle 수) 분포 히스토그램 + 통계 요약."""
    arr = _to_array(arr)
    lengths = np.array([(arr[:, 0] == u).sum() for u in np.unique(arr[:, 0])])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lengths, bins=20, edgecolor="black")
    ax.axvline(lengths.mean(), color="red", linestyle="--", label=f"mean={lengths.mean():.1f}")
    ax.axvline(np.median(lengths), color="green", linestyle="--", label=f"median={np.median(lengths):.1f}")
    ax.set_xlabel("cycles"); ax.set_ylabel("engines"); ax.legend()
    ax.set_title("Engine life distribution")
    plt.tight_layout(); plt.show()
    print(f"engines={len(lengths)}  mean={lengths.mean():.1f}  std={lengths.std():.1f}  "
          f"min={lengths.min()}  max={lengths.max()}")
    return lengths


def plot_sensor_rolling(arr: np.ndarray, unit: int = 1, sensor_idx: int = 2, window: int = 10):
    """한 엔진의 센서 시계열 + 이동평균/이동표준편차."""
    arr = _to_array(arr)
    block = arr[arr[:, 0] == unit]
    cycle = block[:, 1]
    series = pd.Series(block[:, 5 + sensor_idx])
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(cycle, series, alpha=0.5, label="raw")
    axes[0].plot(cycle, roll_mean, color="red", label=f"rolling mean (w={window})")
    axes[0].fill_between(cycle, roll_mean - roll_std, roll_mean + roll_std, alpha=0.2, color="red")
    axes[0].set_title(f"engine {unit} - s{sensor_idx+1}"); axes[0].legend()
    axes[1].plot(cycle, roll_std, color="orange")
    axes[1].set_title("rolling std"); axes[1].set_xlabel("cycle")
    plt.tight_layout(); plt.show()


def decompose_sensor(arr: np.ndarray, unit: int = 1, sensor_idx: int = 2,
                     period: int = 20, model: str = "additive"):
    """이동평균 기반 시계열 분해 (trend / seasonal / residual)."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    arr = _to_array(arr)
    block = arr[arr[:, 0] == unit]
    series = block[:, 5 + sensor_idx]
    result = seasonal_decompose(series, model=model, period=period, extrapolate_trend="freq")
    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(series); axes[0].set_title(f"observed (engine {unit}, s{sensor_idx+1})")
    axes[1].plot(result.trend); axes[1].set_title("trend")
    axes[2].plot(result.seasonal); axes[2].set_title("seasonal")
    axes[3].plot(result.resid); axes[3].set_title("residual"); axes[3].set_xlabel("cycle")
    plt.tight_layout(); plt.show()
    return result


def adf_test_table(arr: np.ndarray, unit: int = 1):
    """엔진 한 대의 21개 센서에 대한 ADF 정상성 검정 결과 테이블."""
    from statsmodels.tsa.stattools import adfuller
    arr = _to_array(arr)
    block = arr[arr[:, 0] == unit]
    rows = []
    for i, name in enumerate(SENSOR_NAMES):
        series = block[:, 5 + i]
        if series.std() < 1e-8:
            rows.append({"sensor": name, "adf_stat": np.nan, "p_value": np.nan, "stationary": "constant"})
            continue
        stat, pvalue, *_ = adfuller(series)
        rows.append({"sensor": name, "adf_stat": stat, "p_value": pvalue,
                      "stationary": "Yes" if pvalue < 0.05 else "No"})
    return pd.DataFrame(rows)


def plot_acf_pacf(arr: np.ndarray, unit: int = 1, sensor_idx: int = 2, lags: int = 40):
    """한 센서의 ACF / PACF 플롯 (자기상관 구조 확인)."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    arr = _to_array(arr)
    block = arr[arr[:, 0] == unit]
    series = block[:, 5 + sensor_idx]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    plot_acf(series, lags=lags, ax=axes[0])
    plot_pacf(series, lags=lags, ax=axes[1], method="ywm")
    axes[0].set_title(f"ACF - s{sensor_idx+1}")
    axes[1].set_title(f"PACF - s{sensor_idx+1}")
    plt.tight_layout(); plt.show()


def plot_sensor_correlation(arr: np.ndarray) -> np.ndarray:
    """21개 센서 상관관계 히트맵."""
    arr = _to_array(arr)
    sensor_data = arr[:, 5:5 + len(SENSOR_NAMES)]
    corr = np.corrcoef(sensor_data.T)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(21)); ax.set_yticks(range(21))
    ax.set_xticklabels(SENSOR_NAMES, rotation=90, fontsize=7)
    ax.set_yticklabels(SENSOR_NAMES, fontsize=7)
    plt.colorbar(im); ax.set_title("21-sensor correlation")
    plt.tight_layout(); plt.show()
    return corr


def sensor_rul_correlation(df: pd.DataFrame):
    """각 센서와 RUL 간 상관계수 막대그래프 (절댓값 기준 정렬). df에 'RUL' 컬럼 필요."""
    corr = df[SENSOR_NAMES + ["RUL"]].corr()["RUL"].drop("RUL")
    out = pd.DataFrame({"sensor": corr.index, "corr_with_RUL": corr.values})
    out = out.reindex(out["corr_with_RUL"].abs().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["crimson" if c < 0 else "steelblue" for c in out["corr_with_RUL"]]
    ax.bar(out["sensor"], out["corr_with_RUL"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("corr with RUL"); ax.set_title("Sensor vs RUL correlation")
    plt.xticks(rotation=90); plt.tight_layout(); plt.show()
    return out


def plot_degradation_curves(arr: np.ndarray, rul: np.ndarray, sensor_idx: int = 2,
                            n_engines: int = 15):
    """여러 엔진의 센서값을 RUL 기준으로 정렬해 평균 열화 곡선을 그림."""
    arr = _to_array(arr)
    units = np.unique(arr[:, 0])[:n_engines]
    fig, ax = plt.subplots(figsize=(8, 5))
    curves = []
    for u in units:
        mask = arr[:, 0] == u
        r, v = rul[mask], arr[mask, 5 + sensor_idx]
        ax.plot(r, v, alpha=0.25, color="gray")
        curves.append((r, v))
    bins = np.arange(0, 130, 5)
    means = []
    for b in bins:
        vals = [v[(r >= b) & (r < b + 5)].mean() for r, v in curves if ((r >= b) & (r < b + 5)).any()]
        means.append(np.mean(vals) if vals else np.nan)
    ax.plot(bins, means, color="red", linewidth=2.5, label="mean degradation")
    ax.invert_xaxis()
    ax.set_xlabel("RUL (remaining cycles)"); ax.set_ylabel(f"s{sensor_idx+1}")
    ax.set_title(f"Degradation pattern - s{sensor_idx+1} (RUL-aligned, {len(units)} engines)")
    ax.legend()
    plt.tight_layout(); plt.show()


def plot_engine_degradation(arr: np.ndarray, rul: np.ndarray, unit: int = 1,
                            sensor_idx: int = 2):
    """선택한 엔진 1대의 열화 곡선(빨강)을 전체 평균(회색 점선)과 비교."""
    arr = _to_array(arr)
    units = np.unique(arr[:, 0])
    bins = np.arange(0, 130, 5)
    means = []
    for b in bins:
        vals = []
        for u in units:
            mask = arr[:, 0] == u
            r, v = rul[mask], arr[mask, 5 + sensor_idx]
            in_bin = (r >= b) & (r < b + 5)
            if in_bin.any():
                vals.append(v[in_bin].mean())
        means.append(np.mean(vals) if vals else np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bins, means, color="gray", linewidth=2, linestyle="--", label="population mean")

    mask = arr[:, 0] == unit
    r, v = rul[mask], arr[mask, 5 + sensor_idx]
    ax.plot(r, v, color="red", linewidth=2, label=f"engine {unit}")

    ax.invert_xaxis()
    ax.set_xlabel("RUL (remaining cycles)"); ax.set_ylabel(f"s{sensor_idx+1}")
    ax.set_title(f"Engine {unit} - s{sensor_idx+1} vs population mean")
    ax.legend()
    plt.tight_layout(); plt.show()


def plot_health_vs_failure_distributions(arr: np.ndarray, rul: np.ndarray,
                                         sensor_idx: int = 2, threshold: int = 20):
    """RUL이 큰(건강) 구간 vs 작은(고장임박) 구간의 센서 분포 비교 + t-검정."""
    from scipy import stats as sstats
    arr = _to_array(arr)
    healthy = arr[rul > 100, 5 + sensor_idx]
    failing = arr[rul < threshold, 5 + sensor_idx]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(healthy, bins=30, alpha=0.5, density=True, label=f"healthy (RUL>100, n={len(healthy)})")
    ax.hist(failing, bins=30, alpha=0.5, density=True, label=f"near failure (RUL<{threshold}, n={len(failing)})")
    ax.set_title(f"s{sensor_idx+1} distribution: healthy vs near-failure")
    ax.legend()
    plt.tight_layout(); plt.show()
    t, p = sstats.ttest_ind(healthy, failing, equal_var=False)
    print(f"Welch t-test: t={t:.3f}, p={p:.2e}")


def normalize_features(X_train: np.ndarray, X_test: Optional[np.ndarray] = None):
    """train 통계로 정규화."""
    mu = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    sd = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-6
    Xt = (X_train - mu) / sd
    if X_test is not None:
        return Xt, (X_test - mu) / sd, mu, sd
    return Xt, mu, sd


# ---------- 모델 ----------

class LSTMRegressor(nn.Module):
    """RUL 예측 baseline."""
    def __init__(self, n_feat: int, hidden: int = 64, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, x):
        # x: (B, T, F)
        out, (h, _) = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class TimeSeriesTransformer(nn.Module):
    """시계열 RUL용 Encoder + mean pooling."""
    def __init__(self, n_feat: int, d_model: int = 64, nhead: int = 4,
                 layers: int = 2, max_len: int = 64):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128,
                                         dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, x):
        h = self.proj(x) + self.pos[:, :x.size(1)]
        h = self.enc(h)
        return self.head(h.mean(dim=1)).squeeze(-1)


def evaluate_rul_predictions(model, X: np.ndarray, y: np.ndarray, device: str = "cpu"):
    """검증 세트 예측 + MSE/MAE/RUL-score/상관계수 계산. Returns (pred, metrics_dict)."""
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X).to(device)).cpu().numpy()
    metrics = {
        "mse": float(np.mean((pred - y) ** 2)),
        "mae": float(np.mean(np.abs(pred - y))),
        "rul_score": rul_score(y, pred),
        "corr": float(np.corrcoef(pred, y)[0, 1]),
    }
    return pred, metrics


def plot_rul_evaluation(losses, vals, y_true: np.ndarray, y_pred: np.ndarray):
    """학습/검증 loss curve + 예측-실제 산점도(상관관계) + 정렬 비교 플롯."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(losses, label="train loss")
    axes[0].plot(vals, label="val loss")
    axes[0].set_title("Loss curve (MSE)"); axes[0].set_xlabel("epoch"); axes[0].legend()

    corr = np.corrcoef(y_pred, y_true)[0, 1]
    axes[1].scatter(y_true, y_pred, alpha=0.3, s=10)
    lo, hi = y_true.min(), y_true.max()
    axes[1].plot([lo, hi], [lo, hi], "r--", label="y = x (perfect)")
    axes[1].set_title(f"Pred vs True (corr={corr:.2f})")
    axes[1].set_xlabel("true RUL"); axes[1].set_ylabel("pred RUL"); axes[1].legend()

    order = np.argsort(y_true)[::-1]
    axes[2].plot(y_true[order], label="true RUL", alpha=0.7)
    axes[2].plot(y_pred[order], label="pred RUL", alpha=0.7)
    axes[2].set_title("Prediction vs True RUL (sorted)")
    axes[2].set_xlabel("sample (sorted by true RUL)"); axes[2].legend()
    plt.tight_layout(); plt.show()


def rul_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """CMAPSS scoring function (asymmetric, late predictions punished more)."""
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(s.sum())


def train_regressor(model, X_tr, y_tr, X_va, y_va, epochs: int = 8,
                    bs: int = 64, lr: float = 1e-3, device: str = "cpu"):
    """RUL 회귀 학습. (loss curve 반환)."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(X_tr).to(device); yt = torch.from_numpy(y_tr).to(device)
    Xv = torch.from_numpy(X_va).to(device); yv = torch.from_numpy(y_va).to(device)
    losses, vals = [], []
    n = Xt.size(0)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pred = model(Xt[idx])
            loss = F.mse_loss(pred, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * idx.size(0)
        ep_loss /= n
        losses.append(ep_loss)
        model.eval()
        with torch.no_grad():
            v = F.mse_loss(model(Xv), yv).item()
        vals.append(v)
    return losses, vals


# ---------------------------------------------------------------
# Part B: ViT helpers
# ---------------------------------------------------------------

class PatchEmbed(nn.Module):
    """이미지 → 패치 토큰. Conv2d로 한번에 split + project."""
    def __init__(self, img_size: int = 32, patch: int = 4, in_ch: int = 3,
                 d_model: int = 96):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)
        self.n_patches = (img_size // patch) ** 2

    def forward(self, x):
        # x: (B, C, H, W) -> (B, N, D)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class TinyViT(nn.Module):
    """CIFAR-10용 작은 ViT."""
    def __init__(self, img_size: int = 32, patch: int = 4, n_classes: int = 10,
                 d_model: int = 96, depth: int = 4, nhead: int = 4):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch, 3, d_model)
        n = self.patch.n_patches
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, n + 1, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 2,
                                         dropout=0.1, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        B = x.size(0)
        h = self.patch(x)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos
        h = self.enc(h)
        h = self.norm(h[:, 0])
        return self.head(h)


class ViTFScratchPatchEmbed(nn.Module):
    """Patch embedding block used in ViT-from-scratch notebook."""
    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, d_model: int = 192):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.n_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        # (B, C, H, W) -> (B, N, D)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class ViTFScratchMHSA(nn.Module):
    """Multi-head self-attention used in ViT-from-scratch notebook."""
    def __init__(self, d_model: int = 192, n_heads: int = 6,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj_drop(self.proj(x))
        return x


class ViTFScratchMLP(nn.Module):
    """MLP block used in ViT-from-scratch notebook."""
    def __init__(self, d_model: int = 192, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, d_model),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class ViTFScratchTransformerBlock(nn.Module):
    """Pre-norm Transformer block used in ViT-from-scratch notebook."""
    def __init__(self, d_model: int = 192, n_heads: int = 6,
                 mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ViTFScratchMHSA(d_model=d_model, n_heads=n_heads, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = ViTFScratchMLP(d_model=d_model, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyCNN(nn.Module):
    """비교용 작은 CNN."""
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 8, 128),
                                nn.ReLU(), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.fc(self.net(x))


def get_cifar10_loaders(bs: int = 128, root: Optional[str] = None,
                        download: bool = True):
    """CIFAR-10 loaders. torchvision."""
    import torchvision
    import torchvision.transforms as T
    if root is None:
        root = str(Path.home() / ".cache" / "cifar10")
    norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    tt = T.Compose([T.ToTensor(), norm])
    tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                    T.ToTensor(), norm])
    tr_ds = torchvision.datasets.CIFAR10(root, train=True, download=download, transform=tr)
    te_ds = torchvision.datasets.CIFAR10(root, train=False, download=download, transform=tt)
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0)
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=bs, shuffle=False, num_workers=0)
    return tr_loader, te_loader, tr_ds.classes


def train_classifier(model, train_loader, test_loader, epochs: int = 3,
                     lr: float = 3e-4, device: str = "cpu"):
    """이미지 분류 학습. loss/acc 기록."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    losses, accs = [], []
    for ep in range(epochs):
        model.train()
        ep_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * x.size(0); n += x.size(0)
        losses.append(ep_loss / n)
        accs.append(evaluate_classifier(model, test_loader, device))
        print(f"ep {ep+1}: loss={losses[-1]:.3f}  acc={accs[-1]:.3f}")
    return losses, accs


def evaluate_classifier(model, loader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=-1)
            correct += (pred == y).sum().item(); total += y.size(0)
    return correct / total


def evaluate_vitfs_classifier(model, loader, device: str = "cpu"):
    """Evaluate ViT-from-scratch model (loss, accuracy)."""
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

    return total_loss / total, total_correct / total


def _train_one_epoch_vitfs(
    model,
    loader,
    optimizer,
    device: str = "cpu",
    grad_clip: Optional[float] = 1.0,
    label_smoothing: float = 0.0,
    max_batches: Optional[int] = None,
):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total


def train_vitfs_classifier(
    model,
    train_loader,
    test_loader,
    cfg: Optional[dict] = None,
    device: str = "cpu",
):
    """Train ViT-from-scratch model with AdamW + cosine scheduler."""
    cfg = cfg or {}
    epochs = int(cfg.get("epochs", 20))
    lr = float(cfg.get("lr", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 0.05))
    label_smoothing = float(cfg.get("label_smoothing", 0.1))
    grad_clip = cfg.get("grad_clip", 1.0)
    max_train_batches = cfg.get("max_train_batches", None)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = defaultdict(list)
    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _train_one_epoch_vitfs(
            model,
            train_loader,
            optimizer,
            device=device,
            grad_clip=grad_clip,
            label_smoothing=label_smoothing,
            max_batches=max_train_batches,
        )
        te_loss, te_acc = evaluate_vitfs_classifier(model, test_loader, device=device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        if te_acc > best_acc:
            best_acc = te_acc
            best_state = copy.deepcopy(model.state_dict())

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:02d}/{epochs} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"test_loss={te_loss:.4f} test_acc={te_acc:.4f}"
        )
        scheduler.step()

    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    print(f"best test acc = {best_acc:.4f}")
    print(f"elapsed time = {elapsed/60:.1f} min")
    return model, dict(history)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_image_grid(images, titles=None, cols: int = 4, figsize=(8, 4)):
    """matplotlib에 이미지 격자."""
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i < n:
            img = images[i]
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            if isinstance(img, np.ndarray):
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = np.transpose(img, (1, 2, 0))
                img = (img - img.min()) / (img.max() - img.min() + 1e-6)
                img = img.squeeze()
            ax.imshow(img, cmap="gray" if (isinstance(img, np.ndarray) and img.ndim == 2) else None)
            if titles is not None and i < len(titles):
                ax.set_title(titles[i], fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    return fig


def denormalize_images(x: torch.Tensor, mean=(0.4914, 0.4822, 0.4465),
                       std=(0.2470, 0.2435, 0.2616)) -> torch.Tensor:
    """Denormalize a CHW/NCHW tensor image batch for visualization."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False
    mean_t = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    out = (x * std_t + mean_t).clamp(0, 1)
    return out[0] if squeeze_back else out


def plot_class_distribution(class_names, counts, title: str = "Class distribution",
                            figsize=(10, 3)):
    """Bar plot for class counts."""
    plt.figure(figsize=figsize)
    plt.bar(class_names, counts, color="steelblue")
    plt.title(title)
    plt.ylabel("count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def get_one_sample_per_class(dataset, class_names):
    """Return one raw image tensor per class from CIFAR-style dataset."""
    targets = np.array(dataset.targets)
    images, titles = [], []
    for class_idx, class_name in enumerate(class_names):
        idx = np.where(targets == class_idx)[0][0]
        img = torch.tensor(dataset.data[idx]).permute(2, 0, 1).float() / 255.0
        images.append(img)
        titles.append(class_name)
    return images, titles


def plot_patch_layout(img_chw: torch.Tensor, patch_size: int = 4, title: str = "Patch layout"):
    """Overlay patch grid lines on one image tensor (C,H,W)."""
    img = img_chw.detach().cpu().permute(1, 2, 0).numpy()
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    for y in range(0, h, patch_size):
        ax.axhline(y - 0.5, color="yellow", linewidth=0.8)
    for x in range(0, w, patch_size):
        ax.axvline(x - 0.5, color="yellow", linewidth=0.8)
    ax.set_title(f"{title} ({patch_size}x{patch_size})")
    ax.axis("off")
    plt.show()


def plot_train_test_curves(history: dict):
    """Plot train/test loss and accuracy curves."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["test_loss"], label="test")
    axes[0].set_title("Loss curve")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["test_acc"], label="test")
    axes[1].set_title("Accuracy curve")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def evaluate_classifier_detailed(model, loader, class_names, device="cpu",
                                 max_errors: int = 20,
                                 mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2470, 0.2435, 0.2616)):
    """Compute confusion matrix/class accuracy and collect misclassified samples."""
    n_classes = len(class_names)
    conf = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    error_images, error_titles = [], []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            pred = logits.argmax(dim=1).cpu()

            for t, p in zip(y, pred):
                conf[t.long(), p.long()] += 1

            wrong = pred != y
            if wrong.any() and len(error_images) < max_errors:
                wrong_x = denormalize_images(x[wrong], mean=mean, std=std)
                wrong_y = y[wrong]
                wrong_p = pred[wrong]

                for img, t, p in zip(wrong_x, wrong_y, wrong_p):
                    if len(error_images) >= max_errors:
                        break
                    error_images.append(img)
                    error_titles.append(f"T:{class_names[t.item()]} | P:{class_names[p.item()]}")

    row_sums = conf.sum(dim=1).clamp(min=1)
    class_acc = conf.diag() / row_sums
    overall_acc = conf.diag().sum().item() / conf.sum().item()
    return conf, class_acc, overall_acc, error_images, error_titles


def plot_confusion_matrix(conf, class_names, title: str = "Confusion Matrix", figsize=(6, 5)):
    """Plot confusion matrix heatmap."""
    conf_np = conf.detach().cpu().numpy() if isinstance(conf, torch.Tensor) else np.asarray(conf)
    plt.figure(figsize=figsize)
    plt.imshow(conf_np, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def class_accuracy_table(class_acc, class_names, sort_desc: bool = True) -> pd.DataFrame:
    """Return class-wise accuracy table."""
    values = class_acc.detach().cpu().numpy() if isinstance(class_acc, torch.Tensor) else np.asarray(class_acc)
    df = pd.DataFrame({"class": class_names, "accuracy": values})
    if sort_desc:
        df = df.sort_values("accuracy", ascending=False)
    return df


# ---------------------------------------------------------------
# CLIP image-text retrieval (color search demo)
# ---------------------------------------------------------------

def load_image_from_url(url: str = "https://picsum.photos/id/237/500/400", timeout: int = 10):
    """Download an RGB image from a URL. Falls back to a synthetic
    color-block image (offline-safe) if the download fails."""
    import ssl
    from PIL import Image
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except urllib.error.URLError as e:
        if isinstance(e.reason, ssl.SSLCertVerificationError):
            # local/corporate proxy with a self-signed cert in the chain
            print("SSL verification failed; retrying without cert verification")
            try:
                ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                    data = resp.read()
            except Exception as e2:
                print(f"image download failed ({e2}); using synthetic placeholder image")
                return _make_placeholder_color_image()
        else:
            print(f"image download failed ({e}); using synthetic placeholder image")
            return _make_placeholder_color_image()
    except Exception as e:
        print(f"image download failed ({e}); using synthetic placeholder image")
        return _make_placeholder_color_image()
    return Image.open(io.BytesIO(data)).convert("RGB")


def _make_placeholder_color_image(size=(500, 400)):
    """Offline fallback: a 2x2 grid of solid color blocks (red/blue/green/yellow)."""
    from PIL import Image, ImageDraw
    w, h = size
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    blocks = {"red": (220, 50, 50), "blue": (50, 90, 220),
              "green": (50, 180, 90), "yellow": (230, 200, 40)}
    cw, ch = w // 2, h // 2
    for i, rgb in enumerate(blocks.values()):
        x0, y0 = (i % 2) * cw, (i // 2) * ch
        draw.rectangle([x0, y0, x0 + cw, y0 + ch], fill=rgb)
    return img


def plot_clip_color_scores(labels, scores, title: str = "CLIP color-search scores", figsize=(7, 3)):
    """Bar chart of CLIP similarity scores per text query, best match highlighted."""
    order = np.argsort(scores)[::-1]
    labels = [labels[i] for i in order]
    scores = [scores[i] for i in order]
    plt.figure(figsize=figsize)
    bars = plt.bar(labels, scores, color="steelblue")
    bars[0].set_color("crimson")
    plt.ylabel("similarity (softmax prob)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def split_image_grid(image, rows: int = 2, cols: int = 2):
    """Split a PIL image into a rows x cols grid of crops (a tiny image gallery)."""
    w, h = image.size
    cw, ch = w // cols, h // rows
    crops = []
    for r in range(rows):
        for c in range(cols):
            box = (c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)
            crops.append(image.crop(box))
    return crops


def show_retrieval_result(crops, scores, query: str, cols: int = 2, figsize=(6, 6)):
    """Show a grid of image crops with CLIP scores, highlighting the best match for `query`."""
    n = len(crops)
    rows = (n + cols - 1) // cols
    best = int(np.argmax(scores))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(crops[i])
            color = "crimson" if i == best else "black"
            weight = "bold" if i == best else "normal"
            ax.set_title(f"score={scores[i]:.3f}", color=color, fontweight=weight)
        ax.axis("off")
    fig.suptitle(f'query: "{query}"  ->  best match: crop #{best}')
    plt.tight_layout()
    plt.show()
    return best


__all__ = [
    "set_seed", "pick_device",
    # CMAPSS
    "synthesize_cmapss", "download_cmapss", "load_cmapss",
    "make_rul_labels", "make_windows", "normalize_features",
    "LSTMRegressor", "TimeSeriesTransformer", "rul_score", "train_regressor",
    "evaluate_rul_predictions", "plot_rul_evaluation",
    "SENSOR_NAMES", "SETTINGS_NAMES", "COLS",
    # EDA
    "describe_sensors", "engine_life_summary", "plot_sensor_rolling",
    "decompose_sensor", "adf_test_table", "plot_acf_pacf",
    "plot_sensor_correlation", "sensor_rul_correlation",
    "plot_degradation_curves", "plot_engine_degradation", "plot_health_vs_failure_distributions",
    # ViT
    "PatchEmbed", "TinyViT", "TinyCNN",
    "ViTFScratchPatchEmbed", "ViTFScratchMHSA", "ViTFScratchMLP", "ViTFScratchTransformerBlock",
    "get_cifar10_loaders", "train_classifier", "evaluate_classifier",
    "evaluate_vitfs_classifier", "train_vitfs_classifier",
    "count_params", "show_image_grid",
    "denormalize_images", "plot_class_distribution", "get_one_sample_per_class",
    "plot_patch_layout", "plot_train_test_curves",
    "evaluate_classifier_detailed", "plot_confusion_matrix", "class_accuracy_table",
    # CLIP retrieval
    "load_image_from_url", "plot_clip_color_scores", "split_image_grid", "show_retrieval_result",
    # libs (re-export so notebooks `from tool4 import *` get torch etc.)
    "torch", "nn", "F", "np", "pd", "plt", "math", "random",
]
