#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAST OOD Flood Prediction (FAST v4)
Key upgrades vs v3:
- Add geo features (lat/lon) + DOY global risk prior
- Always handle class imbalance (sample_weight)
- CORAL and GroupDRO are *candidates* selected by pseudo-OOD val (2 stations)
- Bias-only calibration (delta on logits) under fixed evaluator threshold 0.5
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef


# -------------------------
# Official station split
# -------------------------
TRAINING_STATIONS = [
    'Annapolis','Atlantic_City','Charleston','Washington','Wilmington',
    'Eastport','Portland','Sewells_Point','Sandy_Hook'
]
TESTING_STATIONS = ['Lewes', 'Fernandina_Beach', 'The_Battery']


# -------------------------
# Provided thresholds (predefined)
# -------------------------
PRELOADED_DATA = {
    "station_name": ['Annapolis', 'Atlantic_City', 'Charleston', 'Eastport', 'Fernandina_Beach',
                     'Lewes', 'Portland', 'Sandy_Hook', 'Sewells_Point', 'The_Battery',
                     'Washington', 'Wilmington'],
    "flood_threshold": [2.104, 3.344, 2.98, 8.071, 3.148, 2.675, 6.267, 2.809, 2.706, 3.192, 2.673, 2.423]
}
THR = dict(zip(PRELOADED_DATA["station_name"], PRELOADED_DATA["flood_threshold"]))

HIST_DAYS = 7
FUTURE_DAYS = 14
RNG = np.random.default_rng(42)

# GDRO hyperparams (still small/fast)
GDRO_EPOCHS = 2
GDRO_ETA = 0.6
GDRO_EPS = 1e-6

# CORAL ridge
CORAL_RIDGE = 1e-3


# =========================
# helpers
# =========================
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def qstats(name: str, p: np.ndarray) -> None:
    qs = np.quantile(p, [0.01, 0.10, 0.50, 0.90, 0.99])
    print(f"    {name}: mean={p.mean():.4f} std={p.std():.4f} "
          f"q01={qs[0]:.4f} q10={qs[1]:.4f} q50={qs[2]:.4f} q90={qs[3]:.4f} q99={qs[4]:.4f} "
          f"frac>=0.5={(p>=0.5).mean():.4f}")

def eval_fixed05(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else -1.0
    return {"f1": f1, "acc": acc, "mcc": mcc, "pos": float(y_true.mean()), "pred_pos": float(y_pred.mean())}

def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, dtype=np.float64)
    sig = X.std(axis=0, dtype=np.float64)
    sig = np.where(sig < 1e-6, 1.0, sig)
    return mu.astype(np.float32), sig.astype(np.float32)

def standardize_apply(X: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return ((X - mu) / sig).astype(np.float32)

def logistic_loss_from_logits(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, s) - y.astype(np.float32) * s

def tune_bias_delta(y_val: np.ndarray, s_val: np.ndarray,
                    pos_floor: float = 0.05, pos_ceil: float = 0.60) -> tuple[float, dict]:
    # under fixed evaluator threshold 0.5: sigmoid(s+delta)>=0.5 <=> s+delta>=0
    qs = np.quantile(s_val, np.linspace(0.01, 0.99, 199))
    deltas = -qs

    best = {"delta": 0.0, "score": -1e9, "mcc": -1e9, "f1": -1e9, "acc": -1e9, "pred_pos": 0.0}
    pos = float(y_val.mean())

    for d in deltas:
        y_hat = (s_val + d >= 0.0).astype(int)
        pred_pos = float(y_hat.mean())
        if pred_pos < pos_floor or pred_pos > pos_ceil:
            continue
        if y_hat.sum() in (0, len(y_hat)):
            continue
        mcc = float(matthews_corrcoef(y_val, y_hat))
        f1 = float(f1_score(y_val, y_hat, zero_division=0))
        acc = float(accuracy_score(y_val, y_hat))
        # prefer MCC, keep F1, but discourage crazy pred_pos drift
        score = mcc + 0.35 * f1 - 0.25 * abs(pred_pos - pos)
        if score > best["score"]:
            best = {"delta": float(d), "score": float(score), "mcc": mcc, "f1": f1, "acc": acc, "pred_pos": pred_pos}
    return best["delta"], best


# =========================
# FAST IO + daily aggregation (keep lat/lon)
# =========================
def load_hourly_fast(path: str, allowed_stations: set[str]) -> tuple[pd.DataFrame, int]:
    usecols = ["time", "station_name", "sea_level", "latitude", "longitude"]
    dtypes = {"time": "string", "station_name": "category", "sea_level": "float32",
              "latitude": "float32", "longitude": "float32"}
    df = pd.read_csv(path, usecols=usecols, dtype=dtypes)
    before = len(df)
    df = df.dropna(subset=["sea_level"])
    dropped = before - len(df)
    df = df[df["station_name"].isin(list(allowed_stations))]
    df["date"] = df["time"].str.slice(0, 10)
    return df[["station_name", "date", "sea_level", "latitude", "longitude"]], dropped

def hourly_to_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    daily = (df_hourly.groupby(["station_name", "date"], sort=False, observed=True)
             .agg(sea_level=("sea_level", "mean"),
                  sea_level_max=("sea_level", "max"),
                  latitude=("latitude", "first"),
                  longitude=("longitude", "first"))
             .reset_index())
    daily = daily.sort_values(["station_name", "date"], kind="mergesort").reset_index(drop=True)
    g = daily.groupby("station_name", sort=False, observed=True)
    daily[["sea_level", "sea_level_max"]] = g[["sea_level", "sea_level_max"]].ffill().bfill()
    daily[["sea_level", "sea_level_max"]] = daily[["sea_level", "sea_level_max"]].fillna(0.0)
    # lat/lon constant per station; fill if missing
    daily[["latitude", "longitude"]] = g[["latitude", "longitude"]].ffill().bfill()
    daily[["latitude", "longitude"]] = daily[["latitude", "longitude"]].fillna(0.0)
    return daily


# =========================
# CORAL (cheap for small d)
# =========================
def cov_mat(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    if n <= 1:
        return np.eye(X.shape[1], dtype=np.float32)
    return (X.T @ X) / float(n - 1)

def sqrtm_psd(C: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T

def inv_sqrtm_psd(C: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 0.0, None)
    return (V * (1.0 / np.sqrt(w + 1e-12))) @ V.T

def coral_fit_transform_source(Xs: np.ndarray, Xt: np.ndarray, ridge: float = CORAL_RIDGE) -> tuple[np.ndarray, dict]:
    """
    Return aligned source in target space (preserve target mean):
      Xs' = (Xs - ms) @ A + mt
    Target will be used as-is (Xt).
    """
    ms = Xs.mean(axis=0, keepdims=True)
    mt = Xt.mean(axis=0, keepdims=True)
    Xs0 = Xs - ms
    Xt0 = Xt - mt
    Cs = cov_mat(Xs0) + ridge * np.eye(Xs.shape[1], dtype=np.float32)
    Ct = cov_mat(Xt0) + ridge * np.eye(Xs.shape[1], dtype=np.float32)
    A = inv_sqrtm_psd(Cs) @ sqrtm_psd(Ct)
    Xs_aligned = (Xs0 @ A + mt).astype(np.float32)
    pack = {"ms": ms.astype(np.float32), "mt": mt.astype(np.float32), "A": A.astype(np.float32)}
    return Xs_aligned, pack


# =========================
# Features
# =========================
def build_global_doy_risk(daily_tr: pd.DataFrame) -> pd.Series:
    """
    Build a cheap seasonal prior:
      risk_doy_global[doy] = P(future flood within 14d | doy) aggregated over training stations/years
    """
    tmp = daily_tr.copy()
    tmp["exc_flag"] = (tmp["sea_level_max"] - tmp["station_name"].map(THR).astype("float32") > 0).astype("int8")
    out_rows = []
    for stn, g in tmp.groupby("station_name", sort=False, observed=True):
        f = g["exc_flag"].to_numpy(np.int8)
        if len(f) < FUTURE_DAYS + 2:
            continue
        fw = np.lib.stride_tricks.sliding_window_view(f, FUTURE_DAYS)
        yday = (fw.max(axis=1) > 0).astype(np.int8)
        # align to day index t (start of future window). We'll attach risk to the same date row t.
        # last FUTURE_DAYS-1 days cannot be labeled -> ignore
        doy = g["doy"].to_numpy(np.int16)
        doy = doy[:len(yday)]
        out_rows.append(pd.DataFrame({"doy": doy, "yday": yday}))
    dd = pd.concat(out_rows, ignore_index=True)
    risk = dd.groupby("doy")["yday"].mean()
    return risk

def add_features(daily: pd.DataFrame, risk_doy_global: pd.Series) -> pd.DataFrame:
    daily = daily.copy()
    daily["station_name"] = daily["station_name"].astype(str)
    daily["flood_threshold"] = daily["station_name"].map(THR).astype("float32")

    thr = daily["flood_threshold"].to_numpy(np.float32)
    sl  = daily["sea_level"].to_numpy(np.float32)
    mx  = daily["sea_level_max"].to_numpy(np.float32)

    daily["exc_max"]  = (mx - thr).astype("float32")
    daily["exc_mean"] = (sl - thr).astype("float32")
    daily["surge"]    = (mx - sl).astype("float32")
    daily["exc_flag"] = (daily["exc_max"] > 0.0).astype("float32")

    dt = pd.to_datetime(daily["date"], format="%Y-%m-%d", errors="coerce")
    doy = dt.dt.dayofyear.fillna(1).astype("int16")
    daily["doy"] = doy
    daily["doy_sin"] = np.sin(2*np.pi*doy.to_numpy()/365.25).astype("float32")
    daily["doy_cos"] = np.cos(2*np.pi*doy.to_numpy()/365.25).astype("float32")

    # seasonal risk prior (global)
    daily["doy_risk_global"] = doy.map(risk_doy_global).fillna(risk_doy_global.mean()).astype("float32")

    g = daily.groupby("station_name", sort=False, observed=True)
    for w in (7, 14, 30):
        daily[f"r{w}_mean"] = g["exc_max"].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True).astype("float32")
        daily[f"r{w}_std"]  = g["exc_max"].rolling(w, min_periods=2).std().reset_index(level=0, drop=True).fillna(0.0).astype("float32")
        daily[f"r{w}_max"]  = g["exc_max"].rolling(w, min_periods=1).max().reset_index(level=0, drop=True).astype("float32")
        daily[f"r{w}_cnt"]  = g["exc_flag"].rolling(w, min_periods=1).sum().reset_index(level=0, drop=True).astype("float32")

    daily["d1"] = g["exc_max"].diff().fillna(0.0).astype("float32")

    # geo (OOD cue)
    lat = daily["latitude"].to_numpy(np.float32)
    lon = daily["longitude"].to_numpy(np.float32)
    daily["lat"] = lat
    daily["lon"] = lon
    # periodic encoding for lon, and mild scaling
    daily["lat_sin"] = np.sin(np.deg2rad(lat)).astype("float32")
    daily["lon_sin"] = np.sin(np.deg2rad(lon)).astype("float32")

    # finite safety
    for c in FEAT_COLS:
        daily[c] = daily[c].replace([np.inf, -np.inf], np.nan)
    daily[FEAT_COLS] = daily.groupby("station_name", sort=False, observed=True)[FEAT_COLS].ffill().bfill()
    daily[FEAT_COLS] = daily[FEAT_COLS].fillna(0.0)

    return daily


FEAT_COLS = [
    # core exceedance
    "exc_max","exc_mean","surge",
    # rolling
    "r7_mean","r7_std","r7_max","r7_cnt",
    "r14_mean","r14_std","r14_max","r14_cnt",
    "r30_mean","r30_std","r30_max","r30_cnt",
    # dynamics + season prior
    "d1","doy_sin","doy_cos","doy_risk_global",
    # geo
    "lat","lon","lat_sin","lon_sin"
]


# =========================
# Build samples
# =========================
def build_train_samples_with_group(daily: pd.DataFrame, stations: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    df = daily[daily["station_name"].isin(stations)].copy()
    df = df.sort_values(["station_name", "date"], kind="mergesort").reset_index(drop=True)

    stn_to_gid = {s:i for i,s in enumerate(stations)}
    X_list, y_list, g_list = [], [], []

    for stn, g in df.groupby("station_name", sort=False, observed=True):
        feat = g[FEAT_COLS].to_numpy(np.float32)
        flag = g["exc_flag"].to_numpy(np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        flag = np.nan_to_num(flag, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        n = len(g)
        if n < (HIST_DAYS + FUTURE_DAYS + 1):
            continue

        fw = np.lib.stride_tricks.sliding_window_view(flag, FUTURE_DAYS)
        y = (fw.max(axis=1) > 0).astype(np.int8)

        t_start = HIST_DAYS
        t_end = n - FUTURE_DAYS
        x_idx = np.arange(t_start - 1, t_end, dtype=int)
        y_idx = np.arange(t_start, t_end + 1, dtype=int)

        X_part = feat[x_idx]
        y_part = y[y_idx]
        gid = stn_to_gid[str(stn)]
        g_part = np.full(len(y_part), gid, dtype=np.int16)

        X_list.append(X_part)
        y_list.append(y_part)
        g_list.append(g_part)

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int8)
    g_id = np.concatenate(g_list).astype(np.int16)
    return X, y, g_id, stn_to_gid

def build_test_samples_nearest_prev(daily_te: pd.DataFrame, test_index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    idx = test_index.copy()
    idx["station_name"] = idx["station_name"].astype(str)
    idx["hist_end"] = idx["hist_end"].astype(str)

    d = daily_te.copy()
    d["station_name"] = d["station_name"].astype(str)

    station_data = {}
    for stn, g in d.groupby("station_name", sort=False, observed=True):
        dates = pd.to_datetime(g["date"], format="%Y-%m-%d", errors="coerce").to_numpy(dtype="datetime64[D]")
        feats = g[FEAT_COLS].to_numpy(np.float32)
        order = np.argsort(dates)
        station_data[stn] = (dates[order], feats[order])

    X = np.zeros((len(idx), len(FEAT_COLS)), dtype=np.float32)
    valid = np.ones(len(idx), dtype=np.int8)

    hdates = pd.to_datetime(idx["hist_end"], format="%Y-%m-%d", errors="coerce").to_numpy(dtype="datetime64[D]")
    stns = idx["station_name"].to_numpy()

    for stn in np.unique(stns):
        mask = (stns == stn)
        if stn not in station_data:
            valid[mask] = 0
            continue
        dates, feats = station_data[stn]
        qd = hdates[mask]
        pos = np.searchsorted(dates, qd, side="right") - 1
        ok = pos >= 0
        rows = np.where(mask)[0]
        X[rows[ok]] = feats[pos[ok]]
        valid[rows[~ok]] = 0

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return X, valid


# =========================
# Pseudo-OOD val station selection
# =========================
def pick_val_stations(daily_tr: pd.DataFrame) -> list[str]:
    # choose 2 “most different” stations based on a few shift-sensitive stats (fast)
    feats = ["exc_max", "surge", "r30_max", "r30_cnt", "lat", "lon"]
    rows = []
    for stn, g in daily_tr.groupby("station_name", sort=False, observed=True):
        v = g[feats].to_numpy(np.float32)
        rows.append((stn,
                     float(np.nanmean(v[:,0])), float(np.nanstd(v[:,0])),
                     float(np.nanmean(v[:,1])), float(np.nanstd(v[:,1])),
                     float(np.nanmean(v[:,2])), float(np.nanmean(v[:,3])),
                     float(np.nanmean(v[:,4])), float(np.nanmean(v[:,5]))))
    cols = ["station","exc_mean","exc_std","surge_mean","surge_std","r30_max_mean","r30_cnt_mean","lat","lon"]
    df = pd.DataFrame(rows, columns=cols)
    M = df.drop(columns=["station"]).to_numpy(np.float32)
    mu = M.mean(axis=0, dtype=np.float64)
    dist = np.sqrt(((M - mu) ** 2).sum(axis=1))
    df["dist"] = dist
    df = df.sort_values("dist", ascending=False)
    return df["station"].head(2).tolist()


# =========================
# Training: ERM-balanced and GDRO-balanced
# =========================
def make_sgd() -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=2e-5,
        max_iter=1,
        tol=None,
        average=True,
        random_state=42,
        fit_intercept=True,
    )

def class_weights(y: np.ndarray) -> tuple[float, float]:
    pos = float(y.mean())
    pos = min(max(pos, 1e-4), 1.0 - 1e-4)
    w_pos = 0.5 / pos
    w_neg = 0.5 / (1.0 - pos)
    return w_neg, w_pos  # for y=0, y=1

def erm_train(X: np.ndarray, y: np.ndarray) -> SGDClassifier:
    clf = make_sgd()
    classes = np.array([0, 1], dtype=np.int8)
    w0, w1 = class_weights(y)
    sw = np.where(y == 1, w1, w0).astype(np.float32)
    idx = np.arange(len(y), dtype=np.int64)
    RNG.shuffle(idx)
    clf.partial_fit(X[idx], y[idx], classes=classes, sample_weight=sw[idx])
    return clf

def gdro_train_balanced(X: np.ndarray, y: np.ndarray, g_id: np.ndarray, n_groups: int) -> SGDClassifier:
    clf = make_sgd()
    classes = np.array([0, 1], dtype=np.int8)

    cnt = np.bincount(g_id.astype(np.int64), minlength=n_groups).astype(np.float32)
    inv_cnt = 1.0 / np.maximum(cnt, 1.0)

    w0, w1 = class_weights(y)
    base_sw = np.where(y == 1, w1, w0).astype(np.float32)

    w_g = np.ones(n_groups, dtype=np.float32) / float(n_groups)
    idx_all = np.arange(len(y), dtype=np.int64)

    for ep in range(1, GDRO_EPOCHS + 1):
        RNG.shuffle(idx_all)
        sw = (base_sw * w_g[g_id] * inv_cnt[g_id]).astype(np.float32)
        clf.partial_fit(X[idx_all], y[idx_all], classes=classes, sample_weight=sw[idx_all])

        s = clf.decision_function(X).astype(np.float32)
        loss = logistic_loss_from_logits(y, s).astype(np.float32)

        g_loss = np.zeros(n_groups, dtype=np.float32)
        for g in range(n_groups):
            m = (g_id == g)
            g_loss[g] = float(loss[m].mean()) if m.any() else 0.0

        w_g = w_g * np.exp(GDRO_ETA * g_loss)
        w_g = w_g / (w_g.sum() + GDRO_EPS)

        print(f"    [GDRO ep {ep}/{GDRO_EPOCHS}] group_loss min/mean/max = "
              f"{g_loss.min():.4f}/{g_loss.mean():.4f}/{g_loss.max():.4f} | w_g max={w_g.max():.3f}")

    return clf


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_hourly", default="/home/georgechang/OOD/data/train_hourly.csv")
    ap.add_argument("--test_hourly", default="/home/georgechang/OOD/data/test_hourly.csv")
    ap.add_argument("--test_index", default="/home/georgechang/OOD/data/test_index.csv")
    ap.add_argument("--predictions_out", default="/home/georgechang/OOD/output/predictions.csv")
    args = ap.parse_args()

    print("========== PIPELINE START (FAST v4) ==========")

    print("[1/10] Loading CSVs (FAST: usecols+dtypes, no datetime parse)...")
    tr_h, drop_tr = load_hourly_fast(args.train_hourly, set(TRAINING_STATIONS))
    te_h, drop_te = load_hourly_fast(args.test_hourly, set(TESTING_STATIONS))
    test_index = pd.read_csv(args.test_index)

    print(f"  dropped NaN sea_level rows: train={drop_tr:,}, test={drop_te:,}")
    print(f"  train_hourly loaded rows={len(tr_h):,}, test_hourly loaded rows={len(te_h):,}, test_index rows={len(test_index):,}")

    print("[2/10] Hourly -> Daily aggregation (mean/max + lat/lon) ...")
    daily_tr = hourly_to_daily(tr_h)
    daily_te = hourly_to_daily(te_h)
    print(f"  daily_tr rows={len(daily_tr):,}, daily_te rows={len(daily_te):,}")

    print("[3/10] Build DOY seasonal risk prior from TRAINING stations ...")
    # need doy first
    daily_tr["doy"] = pd.to_datetime(daily_tr["date"], errors="coerce").dt.dayofyear.fillna(1).astype("int16")
    risk_doy_global = build_global_doy_risk(daily_tr)
    print(f"  risk_doy_global: mean={risk_doy_global.mean():.4f}, min={risk_doy_global.min():.4f}, max={risk_doy_global.max():.4f}")

    print("[4/10] Feature engineering (add geo + doy risk) ...")
    daily_tr = add_features(daily_tr, risk_doy_global)
    daily_te = add_features(daily_te, risk_doy_global)
    print(f"  FEAT_COLS={len(FEAT_COLS)}")

    print("[5/10] Build TRAIN samples (+group) ...")
    X, y, g_id, stn_to_gid = build_train_samples_with_group(daily_tr, TRAINING_STATIONS)
    print(f"  X={X.shape}, y_pos_rate={y.mean():.4f}, groups={len(TRAINING_STATIONS)}")

    print("[6/10] Build TEST features (nearest previous date) ...")
    X_te, valid = build_test_samples_nearest_prev(daily_te, test_index)
    print(f"  X_te={X_te.shape}, missing rows={(valid==0).sum():,}")

    print("[7/10] Choose pseudo-OOD VAL stations + split train/val ...")
    val_stations = pick_val_stations(daily_tr)
    val_gids = np.array([stn_to_gid[s] for s in val_stations], dtype=np.int16)
    val_mask = np.isin(g_id, val_gids)
    tr_mask = ~val_mask
    print(f"  chosen pseudo-OOD VAL stations: {val_stations}")
    print(f"  split: X_tr={tr_mask.sum():,}, pos={y[tr_mask].mean():.4f} | X_va={val_mask.sum():,}, pos={y[val_mask].mean():.4f}")

    print("[8/10] Standardize (fit on training-split) ...")
    mu, sig = standardize_fit(X[tr_mask])
    Xs_tr = standardize_apply(X[tr_mask], mu, sig)
    ys_tr = y[tr_mask]
    gs_tr = g_id[tr_mask]

    Xs_va = standardize_apply(X[val_mask], mu, sig)
    ys_va = y[val_mask]

    Xs_te = standardize_apply(X_te, mu, sig)

    # build candidates (very fast)
    candidates = []

    # A) ERM-balanced
    print("  [CAND A] ERM-balanced (no CORAL)...")
    clfA = erm_train(Xs_tr, ys_tr)
    s_va_A = clfA.decision_function(Xs_va).astype(np.float32)
    p_va_A = sigmoid(s_va_A)
    metA = eval_fixed05(ys_va, p_va_A)
    print(f"    VAL@0.5 f1={metA['f1']:.4f} acc={metA['acc']:.4f} mcc={metA['mcc']:.4f} pred_pos={metA['pred_pos']:.4f}")
    candidates.append(("ERM", False, clfA, s_va_A, metA))

    # B) ERM-balanced + CORAL
    print("  [CAND B] ERM-balanced + CORAL...")
    # CORAL uses unlabeled test features (valid only)
    Xt_ref = Xs_te[valid == 1]
    if len(Xt_ref) < 1000:
        Xt_ref = Xs_te
    Xs_tr_c, _ = coral_fit_transform_source(Xs_tr, Xt_ref.astype(np.float32))
    Xs_va_c, _ = coral_fit_transform_source(Xs_va, Xt_ref.astype(np.float32))
    clfB = erm_train(Xs_tr_c, ys_tr)
    s_va_B = clfB.decision_function(Xs_va_c).astype(np.float32)
    p_va_B = sigmoid(s_va_B)
    metB = eval_fixed05(ys_va, p_va_B)
    print(f"    VAL@0.5 f1={metB['f1']:.4f} acc={metB['acc']:.4f} mcc={metB['mcc']:.4f} pred_pos={metB['pred_pos']:.4f}")
    candidates.append(("ERM", True, clfB, s_va_B, metB))

    # C) GDRO-balanced
    print("  [CAND C] GDRO-balanced (no CORAL)...")
    clfC = gdro_train_balanced(Xs_tr, ys_tr, gs_tr, n_groups=len(TRAINING_STATIONS))
    s_va_C = clfC.decision_function(Xs_va).astype(np.float32)
    p_va_C = sigmoid(s_va_C)
    metC = eval_fixed05(ys_va, p_va_C)
    print(f"    VAL@0.5 f1={metC['f1']:.4f} acc={metC['acc']:.4f} mcc={metC['mcc']:.4f} pred_pos={metC['pred_pos']:.4f}")
    candidates.append(("GDRO", False, clfC, s_va_C, metC))

    # D) GDRO-balanced + CORAL
    print("  [CAND D] GDRO-balanced + CORAL...")
    clfD = gdro_train_balanced(Xs_tr_c, ys_tr, gs_tr, n_groups=len(TRAINING_STATIONS))
    s_va_D = clfD.decision_function(Xs_va_c).astype(np.float32)
    p_va_D = sigmoid(s_va_D)
    metD = eval_fixed05(ys_va, p_va_D)
    print(f"    VAL@0.5 f1={metD['f1']:.4f} acc={metD['acc']:.4f} mcc={metD['mcc']:.4f} pred_pos={metD['pred_pos']:.4f}")
    candidates.append(("GDRO", True, clfD, s_va_D, metD))

    # select best by MCC then F1
    candidates.sort(key=lambda x: (x[4]["mcc"], x[4]["f1"]), reverse=True)
    name, use_coral, clf_best, s_va_best, met_best = candidates[0][0], candidates[0][1], candidates[0][2], candidates[0][3], candidates[0][4]
    print(f"[9/10] BEST model = {name} | CORAL={use_coral} => VAL@0.5 mcc={met_best['mcc']:.4f} f1={met_best['f1']:.4f} acc={met_best['acc']:.4f}")

    print("[10/10] Retrain BEST on ALL training data + tune bias delta on pseudo-OOD val + predict TEST ...")
    # Refit standardization on ALL training
    mu_all, sig_all = standardize_fit(X)
    Xs_all = standardize_apply(X, mu_all, sig_all)
    Xs_te_all = standardize_apply(X_te, mu_all, sig_all)

    # recreate val masks for ALL
    val_mask_all = np.isin(g_id, val_gids)
    tr_mask_all = ~val_mask_all

    # unlabeled target ref for CORAL
    Xt_ref_all = Xs_te_all[valid == 1]
    if len(Xt_ref_all) < 1000:
        Xt_ref_all = Xs_te_all

    # train best type
    if use_coral:
        Xs_tr_all_c, _ = coral_fit_transform_source(Xs_all[tr_mask_all], Xt_ref_all.astype(np.float32))
        Xs_va_all_c, _ = coral_fit_transform_source(Xs_all[val_mask_all], Xt_ref_all.astype(np.float32))
        if name == "ERM":
            clf_final = erm_train(Xs_tr_all_c, y[tr_mask_all])
        else:
            clf_final = gdro_train_balanced(Xs_tr_all_c, y[tr_mask_all], g_id[tr_mask_all], n_groups=len(TRAINING_STATIONS))
        s_val = clf_final.decision_function(Xs_va_all_c).astype(np.float32)
        # test features in target space: use standardized as-is (Xt_ref space)
        s_te = clf_final.decision_function(Xs_te_all.astype(np.float32)).astype(np.float32)
    else:
        if name == "ERM":
            clf_final = erm_train(Xs_all[tr_mask_all], y[tr_mask_all])
        else:
            clf_final = gdro_train_balanced(Xs_all[tr_mask_all], y[tr_mask_all], g_id[tr_mask_all], n_groups=len(TRAINING_STATIONS))
        s_val = clf_final.decision_function(Xs_all[val_mask_all]).astype(np.float32)
        s_te = clf_final.decision_function(Xs_te_all).astype(np.float32)

    p_val = sigmoid(s_val)
    qstats("val p_raw", p_val)
    delta, info = tune_bias_delta(y[val_mask_all], s_val)
    print(f"  tuned delta={delta:+.4f} | VAL@0.5 mcc={info['mcc']:.4f} f1={info['f1']:.4f} acc={info['acc']:.4f} pred_pos={info['pred_pos']:.4f}")

    p_te = sigmoid(s_te + delta).astype(np.float32)
    qstats("test p_final", p_te)

    out = pd.DataFrame({"id": test_index["id"].to_numpy(), "y_prob": p_te})
    out.loc[valid == 0, "y_prob"] = 0.5
    out["y_prob"] = out["y_prob"].fillna(0.5).clip(0.0, 1.0)  # required
    out.to_csv(args.predictions_out, index=False)

    print(f"  wrote predictions -> {args.predictions_out} (rows={len(out):,})")
    print(f"  test frac(y_prob>=0.5)={(out['y_prob'].to_numpy()>=0.5).mean():.4f}")
    print("========== PIPELINE END ==========")


if __name__ == "__main__":
    main()



# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# NEUSTG OOD Flood Prediction (v3: anomaly + detrend + faster rolling)

# Evaluator fixed:
#   y_pred = (y_prob >= 0.5).astype(int)

# Main gains in this version:
#   - Add per-station day-of-year climatology anomaly/zscore features.
#   - Add per-station 365-day rolling detrend/zscore features.
#   - Truly vectorize exceedance counts (rolling sum of a boolean flag).
#   - Remove lat/lon to reduce station memorization (OOD across stations).
#   - Keep LOSO CV + calibration selection + logit-shift.
#   - Output safety: fillna(0.5).clip(0,1)

# Run:
#   python3 model.py --train_hourly ... --test_hourly ... --test_index ... --predictions_out ...
# """

# from __future__ import annotations

# import argparse
# from dataclasses import dataclass
# from typing import Dict, List, Tuple

# import numpy as np
# import pandas as pd
# from numpy.lib.stride_tricks import sliding_window_view

# from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
# from sklearn.linear_model import LogisticRegression
# from sklearn.isotonic import IsotonicRegression

# # -------------------------
# # Official station split
# # -------------------------
# TRAINING_STATIONS = [
#     'Annapolis','Atlantic_City','Charleston','Washington','Wilmington',
#     'Eastport','Portland','Sewells_Point','Sandy_Hook'
# ]
# TESTING_STATIONS = ['Lewes', 'Fernandina_Beach', 'The_Battery']

# # -------------------------
# # Preloaded thresholds (given)
# # -------------------------
# PRELOADED_DATA = {
#     "station_name": [
#         'Annapolis', 'Atlantic_City', 'Charleston', 'Eastport', 'Fernandina_Beach',
#         'Lewes', 'Portland', 'Sandy_Hook', 'Sewells_Point', 'The_Battery',
#         'Washington', 'Wilmington'
#     ],
#     "flood_threshold": [2.104, 3.344, 2.98, 8.071, 3.148, 2.675, 6.267, 2.809, 2.706, 3.192, 2.673, 2.423]
# }
# THRESHOLDS_DF = pd.DataFrame(PRELOADED_DATA)

# # -------------------------
# # Constants
# # -------------------------
# HIST_DAYS = 7
# FUTURE_DAYS = 14
# EPS = 1e-8
# RNG = np.random.default_rng(42)
# MAX_TRAIN_SAMPLES = 220_000

# # -------------------------
# # Model backend
# # -------------------------
# try:
#     from xgboost import XGBClassifier
#     HAS_XGB = True
# except Exception:
#     HAS_XGB = False
#     from sklearn.ensemble import HistGradientBoostingClassifier


# # =========================
# # Metrics / helpers
# # =========================
# def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int,int,int,int]:
#     tp = int(((y_true == 1) & (y_pred == 1)).sum())
#     tn = int(((y_true == 0) & (y_pred == 0)).sum())
#     fp = int(((y_true == 0) & (y_pred == 1)).sum())
#     fn = int(((y_true == 1) & (y_pred == 0)).sum())
#     return tp, tn, fp, fn

# def eval_fixed05(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
#     y_pred = (y_prob >= 0.5).astype(int)
#     f1 = float(f1_score(y_true, y_pred, zero_division=0))
#     acc = float(accuracy_score(y_true, y_pred))
#     mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else -1.0
#     tp, tn, fp, fn = confusion_counts(y_true, y_pred)
#     return {
#         "f1": f1, "acc": acc, "mcc": mcc,
#         "pos_rate": float(y_true.mean()),
#         "pred_pos_rate": float(y_pred.mean()),
#         "tp": tp, "tn": tn, "fp": fp, "fn": fn
#     }

# def pick_t_by_mcc(y_true: np.ndarray, p: np.ndarray) -> float:
#     u = np.unique(p)
#     if u.size > 2500:
#         u = np.quantile(p, np.linspace(0.01, 0.99, 199))
#     best_t, best_mcc, best_f1 = 0.5, -1e9, -1e9
#     for t in u:
#         y_hat = (p >= t).astype(int)
#         if y_hat.sum() in (0, len(y_hat)):
#             mcc = -1e9
#         else:
#             mcc = matthews_corrcoef(y_true, y_hat)
#         f1 = f1_score(y_true, y_hat, zero_division=0)
#         if (mcc > best_mcc) or (mcc == best_mcc and f1 > best_f1):
#             best_t, best_mcc, best_f1 = float(t), float(mcc), float(f1)
#     return best_t

# def logit(p: np.ndarray) -> np.ndarray:
#     p = np.clip(p, 1e-6, 1 - 1e-6)
#     return np.log(p / (1 - p))

# def sigmoid(z: np.ndarray) -> np.ndarray:
#     return 1.0 / (1.0 + np.exp(-z))

# def logit_shift(p: np.ndarray, t_star: float) -> np.ndarray:
#     p = np.clip(p, 1e-6, 1 - 1e-6)
#     t_star = float(np.clip(t_star, 1e-6, 1 - 1e-6))
#     return sigmoid(logit(p) - logit(np.array([t_star], dtype=p.dtype)))

# def describe_probs(name: str, p: np.ndarray):
#     q = np.quantile(p, [0.01, 0.1, 0.5, 0.9, 0.99])
#     print(f"    {name}: mean={p.mean():.4f} std={p.std():.4f} "
#           f"q01={q[0]:.4f} q10={q[1]:.4f} q50={q[2]:.4f} q90={q[3]:.4f} q99={q[4]:.4f} "
#           f"frac>=0.5={(p>=0.5).mean():.4f}")


# # =========================
# # Daily features (threshold-relative + anomaly + detrend)
# # =========================
# def daily_aggregate_with_threshold(hourly: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
#     df = hourly.copy()
#     df["time"] = pd.to_datetime(df["time"], utc=False)
#     df["date"] = df["time"].dt.floor("D")

#     daily = (df.groupby(["station_name", "date"], sort=False)
#                .agg(sea_level=("sea_level", "mean"),
#                     sea_level_max=("sea_level", "max"))
#                .reset_index())

#     daily = daily.merge(thresholds, on="station_name", how="left")
#     if daily["flood_threshold"].isna().any():
#         miss = daily[daily["flood_threshold"].isna()]["station_name"].unique().tolist()
#         raise RuntimeError(f"Missing thresholds for stations: {miss}")

#     daily = daily.sort_values(["station_name", "date"], kind="mergesort").reset_index(drop=True)
#     g = daily.groupby("station_name", sort=False)

#     thr = daily["flood_threshold"].to_numpy(dtype=np.float32)
#     sl = daily["sea_level"].to_numpy(dtype=np.float32)
#     mx = daily["sea_level_max"].to_numpy(dtype=np.float32)

#     # Station-normalized exceedance (key for OOD)
#     daily["exc_mean"] = sl - thr
#     daily["exc_max"]  = mx - thr
#     daily["exc_mean_ratio"] = (sl / (thr + EPS)) - 1.0
#     daily["exc_max_ratio"]  = (mx / (thr + EPS)) - 1.0

#     daily["surge"] = mx - sl
#     daily["exc_flag"] = (daily["exc_max"] > 0.0).astype(np.float32)

#     # Diffs
#     daily["d1_exc"] = g["exc_max"].diff().fillna(0.0)
#     daily["d3_exc"] = g["exc_max"].diff(3).fillna(0.0)
#     daily["d7_exc"] = g["exc_max"].diff(7).fillna(0.0)

#     # Rolling stats (multi-scale)
#     for w in (3, 7, 14, 28):
#         daily[f"exc_mean_{w}"] = (g["exc_max"].rolling(w, min_periods=1).mean()
#                                  .reset_index(level=0, drop=True))
#         daily[f"exc_std_{w}"] = (g["exc_max"].rolling(w, min_periods=2).std()
#                                 .reset_index(level=0, drop=True)).fillna(0.0)
#         daily[f"exc_max_{w}"] = (g["exc_max"].rolling(w, min_periods=1).max()
#                                  .reset_index(level=0, drop=True))
#         # FAST count of exceedance days (vectorized)
#         daily[f"exc_cnt_{w}"] = (g["exc_flag"].rolling(w, min_periods=1).sum()
#                                  .reset_index(level=0, drop=True)).astype(np.float32)

#     # --- NEW: day-of-year climatology anomaly (station-wise)
#     daily["doy"] = daily["date"].dt.dayofyear.astype(np.int16)
#     # mean/std by (station, doy) computed from available history for that station
#     key = ["station_name", "doy"]
#     doy_mean = daily.groupby(key, sort=False)["exc_max"].transform("mean").astype(np.float32)
#     doy_std  = daily.groupby(key, sort=False)["exc_max"].transform("std").astype(np.float32).fillna(0.0)
#     daily["exc_anom_doy"] = (daily["exc_max"] - doy_mean).astype(np.float32)
#     daily["exc_z_doy"] = (daily["exc_anom_doy"] / (doy_std + 0.05)).astype(np.float32)  # small floor for stability

#     # --- NEW: long-term detrend (365-day rolling mean/std per station)
#     roll_mean_365 = (g["exc_max"].rolling(365, min_periods=30).mean()
#                      .reset_index(level=0, drop=True)).astype(np.float32)
#     roll_std_365  = (g["exc_max"].rolling(365, min_periods=30).std()
#                      .reset_index(level=0, drop=True)).astype(np.float32).fillna(0.0)
#     daily["exc_detr_365"] = (daily["exc_max"] - roll_mean_365).fillna(0.0).astype(np.float32)
#     daily["exc_z_365"] = (daily["exc_detr_365"] / (roll_std_365 + 0.05)).astype(np.float32)

#     # Seasonality Fourier (still helpful)
#     doy = daily["doy"].to_numpy()
#     daily["doy_sin1"] = np.sin(2*np.pi*doy/365.25)
#     daily["doy_cos1"] = np.cos(2*np.pi*doy/365.25)
#     daily["doy_sin2"] = np.sin(4*np.pi*doy/365.25)
#     daily["doy_cos2"] = np.cos(4*np.pi*doy/365.25)

#     return daily


# FEATURES = [
#     "exc_mean","exc_max","exc_mean_ratio","exc_max_ratio",
#     "surge","d1_exc","d3_exc","d7_exc",
#     "exc_mean_3","exc_std_3","exc_max_3","exc_cnt_3",
#     "exc_mean_7","exc_std_7","exc_max_7","exc_cnt_7",
#     "exc_mean_14","exc_std_14","exc_max_14","exc_cnt_14",
#     "exc_mean_28","exc_std_28","exc_max_28","exc_cnt_28",
#     # NEW anomaly/detrend
#     "exc_anom_doy","exc_z_doy","exc_detr_365","exc_z_365",
#     # seasonal Fourier
#     "doy_sin1","doy_cos1","doy_sin2","doy_cos2",
# ]


# # =========================
# # Window features
# # =========================
# def _window_slope(z: np.ndarray) -> np.ndarray:
#     H = z.shape[1]
#     t = np.arange(H, dtype=np.float32)
#     t = (t - t.mean()) / (t.std() + EPS)
#     return np.mean(t[None, :, None] * z, axis=1)

# def make_window_features(hist: np.ndarray) -> np.ndarray:
#     mu = hist.mean(axis=1, keepdims=True)
#     sd = hist.std(axis=1, keepdims=True) + EPS
#     z = (hist - mu) / sd
#     flat = z.reshape(z.shape[0], -1).astype(np.float32)

#     mean = hist.mean(axis=1)
#     std = hist.std(axis=1)
#     mn = hist.min(axis=1)
#     mx = hist.max(axis=1)
#     last = hist[:, -1, :]
#     slope = _window_slope(z)

#     summ = np.concatenate([mean, std, mn, mx, last, slope], axis=1).astype(np.float32)
#     return np.concatenate([flat, summ], axis=1)


# # =========================
# # Window builders
# # =========================
# @dataclass
# class StationWindows:
#     X: np.ndarray
#     y: np.ndarray
#     n: int
#     pos_rate: float

# def build_station_windows(daily: pd.DataFrame, stations: List[str]) -> Dict[str, StationWindows]:
#     df = daily[daily["station_name"].isin(stations)].copy()
#     df["flood_day"] = (df["exc_max"] > 0.0).astype(np.int8)

#     out: Dict[str, StationWindows] = {}
#     F = len(FEATURES)

#     for stn, g in df.groupby("station_name", sort=False):
#         g = g.sort_values("date", kind="mergesort").reset_index(drop=True)
#         if len(g) < HIST_DAYS + FUTURE_DAYS:
#             continue

#         feat = g[FEATURES].to_numpy(dtype=np.float32)
#         if np.isnan(feat).any():
#             feat = pd.DataFrame(feat).ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)

#         flood = g["flood_day"].to_numpy(dtype=np.int8)

#         hist = sliding_window_view(feat, window_shape=(HIST_DAYS, F))[:, 0, :, :]
#         fut = sliding_window_view(flood[HIST_DAYS:], window_shape=FUTURE_DAYS)
#         y = (fut.max(axis=1) > 0).astype(np.int8)

#         n = min(hist.shape[0], y.shape[0])
#         X = make_window_features(hist[:n])
#         y = y[:n]

#         out[stn] = StationWindows(X=X, y=y, n=int(n), pos_rate=float(y.mean()))
#     return out

# def build_test_windows_aligned(daily_te: pd.DataFrame, index: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
#     idx = index.copy()
#     idx["hist_start"] = pd.to_datetime(idx["hist_start"])
#     idx["future_start"] = pd.to_datetime(idx["future_start"])

#     X_list, row_list = [], []
#     F = len(FEATURES)

#     for stn, g in daily_te.groupby("station_name", sort=False):
#         if stn not in TESTING_STATIONS:
#             continue
#         need = idx[idx["station_name"] == stn]
#         if need.empty:
#             continue

#         g = g.sort_values("date", kind="mergesort").reset_index(drop=True)
#         feat = g[FEATURES].to_numpy(dtype=np.float32)
#         if np.isnan(feat).any():
#             feat = pd.DataFrame(feat).ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)

#         pos = pd.Series(np.arange(len(g), dtype=int), index=pd.to_datetime(g["date"]).values)
#         i_start = pos.reindex(need["hist_start"].values).to_numpy()
#         i_fut = pos.reindex(need["future_start"].values).to_numpy()

#         valid = (~np.isnan(i_start)) & (~np.isnan(i_fut)) & ((i_start + HIST_DAYS) == i_fut)
#         if valid.sum() == 0:
#             continue

#         starts = i_start[valid].astype(int)
#         idx_mat = starts[:, None] + np.arange(HIST_DAYS)[None, :]
#         hist = feat[idx_mat]

#         X_list.append(make_window_features(hist))
#         row_list.append(need.index.to_numpy()[valid])

#     if not X_list:
#         raise RuntimeError("No test windows built. Check date alignment.")
#     return np.vstack(X_list), np.concatenate(row_list)


# # =========================
# # Training helpers
# # =========================
# def downsample_keep_all_pos(X: np.ndarray, y: np.ndarray, max_n: int) -> Tuple[np.ndarray, np.ndarray]:
#     if len(y) <= max_n:
#         return X, y
#     pos_idx = np.where(y == 1)[0]
#     neg_idx = np.where(y == 0)[0]
#     remain = max_n - len(pos_idx)
#     if remain <= 0:
#         sel = RNG.choice(pos_idx, size=max_n, replace=False)
#     else:
#         keep_neg = RNG.choice(neg_idx, size=min(remain, len(neg_idx)), replace=False)
#         sel = np.concatenate([pos_idx, keep_neg])
#     RNG.shuffle(sel)
#     return X[sel], y[sel]

# def train_model(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
#     pos = int((y == 1).sum())
#     neg = int((y == 0).sum())
#     spw = float(neg / max(pos, 1))

#     if HAS_XGB:
#         clf = XGBClassifier(
#             random_state=42,
#             n_estimators=2500,
#             max_depth=5,
#             learning_rate=0.03,
#             subsample=0.85,
#             colsample_bytree=0.85,
#             reg_lambda=1.0,
#             min_child_weight=3,
#             gamma=0.0,
#             max_delta_step=1,
#             objective="binary:logistic",
#             eval_metric="aucpr",  # better aligned with imbalance
#             n_jobs=-1,
#             scale_pos_weight=spw,
#             tree_method="hist",
#             early_stopping_rounds=120
#         )
#         clf.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
#         return clf
#     else:
#         clf = HistGradientBoostingClassifier(
#             random_state=42,
#             max_depth=10,
#             learning_rate=0.05,
#             max_iter=900,
#         )
#         clf.fit(X, y)
#         return clf


# # =========================
# # Calibration candidates (auto-select)
# # =========================
# class Calibrator:
#     def fit(self, p: np.ndarray, y: np.ndarray): ...
#     def transform(self, p: np.ndarray) -> np.ndarray: ...

# class IdentityCalibrator(Calibrator):
#     def fit(self, p: np.ndarray, y: np.ndarray): return self
#     def transform(self, p: np.ndarray) -> np.ndarray: return np.clip(p, 0.0, 1.0)

# class PlattOnP(Calibrator):
#     def __init__(self):
#         self.lr = LogisticRegression(solver="lbfgs", max_iter=2000)
#     def fit(self, p: np.ndarray, y: np.ndarray):
#         self.lr.fit(p.reshape(-1,1), y)
#         return self
#     def transform(self, p: np.ndarray) -> np.ndarray:
#         return self.lr.predict_proba(p.reshape(-1,1))[:,1]

# class PlattOnLogitP(Calibrator):
#     def __init__(self):
#         self.lr = LogisticRegression(solver="lbfgs", max_iter=2000)
#     def fit(self, p: np.ndarray, y: np.ndarray):
#         self.lr.fit(logit(p).reshape(-1,1), y)
#         return self
#     def transform(self, p: np.ndarray) -> np.ndarray:
#         return self.lr.predict_proba(logit(p).reshape(-1,1))[:,1]

# class IsotonicCalibrator(Calibrator):
#     def __init__(self):
#         self.iso = IsotonicRegression(out_of_bounds="clip")
#     def fit(self, p: np.ndarray, y: np.ndarray):
#         self.iso.fit(np.clip(p, 1e-6, 1-1e-6), y)
#         return self
#     def transform(self, p: np.ndarray) -> np.ndarray:
#         return self.iso.transform(np.clip(p, 1e-6, 1-1e-6))

# def select_best_calibrator(oof_p_raw: np.ndarray, y: np.ndarray) -> Tuple[Calibrator, float, dict]:
#     cands = [
#         ("identity", IdentityCalibrator()),
#         ("platt_p", PlattOnP()),
#         ("platt_logit", PlattOnLogitP()),
#         ("isotonic", IsotonicCalibrator()),
#     ]

#     best = None
#     for name, cal in cands:
#         cal.fit(oof_p_raw, y)
#         p_cal = cal.transform(oof_p_raw).astype(np.float32)
#         t_star = pick_t_by_mcc(y, p_cal)
#         p_shift = logit_shift(p_cal, t_star).astype(np.float32)
#         m = eval_fixed05(y, p_shift)
#         print(f"  [CALIB {name:10s}] t_star={t_star:.4f}  "
#               f"OOF@0.5 f1={m['f1']:.4f} acc={m['acc']:.4f} mcc={m['mcc']:.4f} "
#               f"pred_pos={m['pred_pos_rate']:.4f} fp={m['fp']} fn={m['fn']}")
#         if best is None:
#             best = (name, cal, t_star, m)
#         else:
#             _, _, _, bm = best
#             if (m["mcc"] > bm["mcc"]) or (m["mcc"] == bm["mcc"] and m["f1"] > bm["f1"]):
#                 best = (name, cal, t_star, m)

#     name, cal, t_star, m = best
#     print(f"  => BEST calibrator={name}, t_star={t_star:.4f}, OOF@0.5 mcc={m['mcc']:.4f}, f1={m['f1']:.4f}")
#     return cal, t_star, m


# # =========================
# # Main
# # =========================
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--train_hourly", default="/home/georgechang/OOD/data/train_hourly.csv")
#     ap.add_argument("--test_hourly", default="/home/georgechang/OOD/data/test_hourly.csv")
#     ap.add_argument("--test_index", default="/home/georgechang/OOD/data/test_index.csv")
#     ap.add_argument("--predictions_out", default="/home/georgechang/OOD/output/predictions.csv")
#     args = ap.parse_args()

#     print("========== PIPELINE START ==========")
#     print("[1/9] Loading CSVs...")
#     train = pd.read_csv(args.train_hourly)
#     test = pd.read_csv(args.test_hourly)
#     test_index = pd.read_csv(args.test_index)
#     print(f"  train_hourly rows={len(train):,}, test_hourly rows={len(test):,}, test_index rows={len(test_index):,}")

#     print("[2/9] Filtering stations strictly by official lists...")
#     train = train[train["station_name"].isin(TRAINING_STATIONS)].copy()
#     test = test[test["station_name"].isin(TESTING_STATIONS)].copy()
#     test_index = test_index[test_index["station_name"].isin(TESTING_STATIONS)].copy()
#     print(f"  filtered train_hourly rows={len(train):,}, filtered test_hourly rows={len(test):,}, filtered test_index rows={len(test_index):,}")
#     print(f"  TRAIN stations found: {sorted(train['station_name'].unique().tolist())}")
#     print(f"  TEST stations found: {sorted(test['station_name'].unique().tolist())}")

#     print("[3/9] Daily aggregation + exceedance + anomaly/detrend features...")
#     daily_tr = daily_aggregate_with_threshold(train, THRESHOLDS_DF)
#     daily_te = daily_aggregate_with_threshold(test, THRESHOLDS_DF)
#     print(f"  daily_tr rows={len(daily_tr):,}, daily_te rows={len(daily_te):,}")
#     print("  daily_tr date range:", str(daily_tr['date'].min()), "->", str(daily_tr['date'].max()))
#     print("  daily_te date range:", str(daily_te['date'].min()), "->", str(daily_te['date'].max()))

#     print("[4/9] Build per-station windows (vectorized) ...")
#     stn_windows = build_station_windows(daily_tr, TRAINING_STATIONS)
#     for s in TRAINING_STATIONS:
#         w = stn_windows.get(s)
#         if w is None:
#             print(f"  WARN: no windows for {s}")
#         else:
#             print(f"  {s:15s} n={w.n:7d} pos_rate={w.pos_rate:.4f}")

#     # concatenate and station slices
#     slices: Dict[str, slice] = {}
#     X_all_list, y_all_list = [], []
#     cur = 0
#     for s in TRAINING_STATIONS:
#         if s not in stn_windows:
#             continue
#         Xs, ys = stn_windows[s].X, stn_windows[s].y
#         X_all_list.append(Xs); y_all_list.append(ys)
#         slices[s] = slice(cur, cur + len(ys))
#         cur += len(ys)

#     X_all = np.vstack(X_all_list)
#     y_all = np.concatenate(y_all_list).astype(np.int8)
#     base_rate = float(y_all.mean())
#     print(f"  TOTAL train windows: X_all={X_all.shape}, pos_rate={base_rate:.4f}")

#     print("[5/9] LOSO OOD-style CV ...")
#     oof_p_raw = np.zeros(len(y_all), dtype=np.float32)
#     holdouts = [s for s in TRAINING_STATIONS if s in stn_windows]

#     for fold, holdout in enumerate(holdouts, start=1):
#         val = stn_windows[holdout]
#         train_stations = [s for s in holdouts if s != holdout]
#         X_tr = np.vstack([stn_windows[s].X for s in train_stations])
#         y_tr = np.concatenate([stn_windows[s].y for s in train_stations]).astype(np.int8)

#         X_tr, y_tr = downsample_keep_all_pos(X_tr, y_tr, MAX_TRAIN_SAMPLES)

#         print(f"\n  [FOLD {fold:02d}/{len(holdouts)}] holdout={holdout}")
#         print(f"    train X={X_tr.shape}, pos_rate={y_tr.mean():.4f} | val X={val.X.shape}, pos_rate={val.y.mean():.4f}")

#         clf = train_model(X_tr, y_tr, val.X, val.y)
#         p_val = clf.predict_proba(val.X)[:, 1].astype(np.float32)

#         describe_probs("p_val_raw", p_val)
#         m_raw = eval_fixed05(val.y, p_val)
#         print(f"    RAW@0.5  f1={m_raw['f1']:.4f} acc={m_raw['acc']:.4f} mcc={m_raw['mcc']:.4f} "
#               f"tp={m_raw['tp']} tn={m_raw['tn']} fp={m_raw['fp']} fn={m_raw['fn']}")

#         oof_p_raw[slices[holdout]] = p_val

#     print("\n[6/9] Calibrator selection + logit-shift ...")
#     describe_probs("OOF p_raw", oof_p_raw)
#     cal, t_star, _ = select_best_calibrator(oof_p_raw, y_all)

#     print("[7/9] Train final model on ALL training stations ...")
#     idx = np.arange(len(y_all))
#     RNG.shuffle(idx)
#     cut = max(int(0.02 * len(idx)), 1)
#     val_idx = idx[:cut]
#     tr_idx = idx[cut:]
#     clf_final = train_model(X_all[tr_idx], y_all[tr_idx], X_all[val_idx], y_all[val_idx])

#     print("[8/9] Build TEST windows aligned to test_index and predict ...")
#     X_te, row_ids = build_test_windows_aligned(daily_te, test_index)
#     print(f"  test windows built: X_te={X_te.shape}, mapped rows={len(row_ids):,}/{len(test_index):,}")

#     p_raw_te = clf_final.predict_proba(X_te)[:, 1].astype(np.float32)
#     describe_probs("test p_raw", p_raw_te)

#     p_cal_te = cal.transform(p_raw_te).astype(np.float32)
#     describe_probs("test p_cal", p_cal_te)

#     p_out = logit_shift(p_cal_te, t_star).astype(np.float32)
#     describe_probs("test p_final", p_out)

#     print("[9/9] Write predictions (safety fill + clip) ...")
#     y_prob_full = np.full(len(test_index), 0.5, dtype=np.float32)  # per your request
#     y_prob_full[row_ids] = np.clip(p_out, 0.0, 1.0)

#     out = pd.DataFrame({"id": test_index["id"].to_numpy(), "y_prob": y_prob_full})
#     # required safety line
#     out["y_prob"] = out["y_prob"].fillna(0.5).clip(0.0, 1.0)

#     out.to_csv(args.predictions_out, index=False)
#     print(f"Wrote predictions -> {args.predictions_out} (rows={len(out):,})")
#     print("========== PIPELINE END ==========")


# if __name__ == "__main__":
#     main()



# #!/usr/bin/env python3
# """
# Participant code-submission baseline - Preloaded & Robust Version.
# Usage:
# python -m model --train_hourly <csv> --test_hourly <csv> --test_index <csv> --predictions_out <csv>
# """
# import argparse
# import pandas as pd
# import numpy as np

# # Note: scipy.io is no longer needed since data is preloaded

# try:
#     from xgboost import XGBClassifier
#     _HAS_XGB = True
# except Exception:
#     from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
#     _HAS_XGB = False

# HIST_DAYS = 7
# FUTURE_DAYS = 14
# FEATURES = [
#     "sea_level", 
#     "sea_level_diff", 
#     "sea_level_3d_mean", 
#     "sea_level_7d_mean",
#     "sea_level_3d_std", 
#     "sea_level_7d_std", 
#     "sin_doy", 
#     "cos_doy"
# ]

# # ==========================================
# # [SECTION TO PASTE YOUR DATA]
# # ==========================================
# # Replace the dictionary below with the output from the Helper Script.
# PRELOADED_DATA = {
#     "station_name": ['Annapolis', 'Atlantic_City', 'Charleston', 'Eastport', 'Fernandina_Beach', 'Lewes', 'Portland', 'Sandy_Hook', 'Sewells_Point', 'The_Battery', 'Washington', 'Wilmington'],
#     "flood_threshold": [2.104, 3.344, 2.98, 8.071, 3.148, 2.675, 6.267, 2.809, 2.706, 3.192, 2.673, 2.423]
# }
# # ==========================================

# def get_preloaded_thresholds():
#     """Returns the thresholds from the hardcoded dictionary."""
#     df_thr = pd.DataFrame(PRELOADED_DATA)
#     if df_thr.empty:
#         raise ValueError("Preloaded threshold data is empty. Please fill in PRELOADED_DATA.")
#     print(f"Loaded {len(df_thr)} stations from preloaded data.")
#     return df_thr

# def daily_aggregate_vectorized(df):
#     """
#     Robust aggregation that reindexes dates to handle missing days (gaps).
#     """
#     df = df.copy()
#     df["time"] = pd.to_datetime(df["time"])
#     df["date"] = df["time"].dt.floor("D")
    
#     # 1. Basic Aggregation
#     daily = (df.groupby(["station_name", "date"])
#                .agg(sea_level=("sea_level", "mean"),
#                     sea_level_max=("sea_level", "max"),
#                     latitude=("latitude", "first"),
#                     longitude=("longitude", "first"))
#                .reset_index())

#     # 2. Reindex to Fill Missing Dates (Fixes missing predictions)
#     min_date = daily["date"].min()
#     max_date = daily["date"].max()
#     all_dates = pd.date_range(min_date, max_date, freq='D')
    
#     mux = pd.MultiIndex.from_product([daily["station_name"].unique(), all_dates], 
#                                      names=["station_name", "date"])
    
#     daily = daily.set_index(["station_name", "date"]).reindex(mux).reset_index()
#     daily = daily.sort_values(["station_name", "date"]).reset_index(drop=True)
    
#     # 3. Imputation (Fill Gaps)
#     daily["sea_level"] = daily.groupby("station_name")["sea_level"].ffill(limit=3)
#     daily["sea_level_max"] = daily.groupby("station_name")["sea_level_max"].ffill(limit=3)
    
#     # Fallback for remaining NaNs
#     daily["sea_level"] = daily["sea_level"].fillna(daily.groupby("station_name")["sea_level"].transform("mean"))
#     daily["latitude"] = daily.groupby("station_name")["latitude"].ffill().bfill()
#     daily["longitude"] = daily.groupby("station_name")["longitude"].ffill().bfill()

#     # 4. Feature Engineering
#     day_of_year = daily["date"].dt.dayofyear
#     daily["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25)
#     daily["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25)

#     grouped = daily.groupby("station_name")["sea_level"]
    
#     daily["sea_level_3d_mean"] = grouped.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
#     daily["sea_level_7d_mean"] = grouped.rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
#     daily["sea_level_3d_std"] = grouped.rolling(3, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
#     daily["sea_level_7d_std"] = grouped.rolling(7, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
#     daily["sea_level_diff"] = grouped.diff().fillna(0)
    
#     return daily

# def build_features_vectorized(daily, stations, use_labels=False, thresholds=None):
#     # Filter stations
#     df = daily[daily["station_name"].isin(stations)].copy()
#     df = df.sort_values(["station_name", "date"]).reset_index(drop=True)

#     if use_labels:
#         if thresholds is None:
#             raise ValueError("Training mode requires thresholds.")
#         df = df.merge(thresholds, on="station_name", how="left")
#         if df["flood_threshold"].isna().any():
#             df = df.dropna(subset=["flood_threshold"])
#         df["flood"] = (df["sea_level_max"] > df["flood_threshold"]).astype(int)

#     # --- Construct Features (X) ---
#     feature_cols = []
    
#     for k in range(HIST_DAYS):
#         shifted = df[FEATURES].shift(-k)
#         shifted.columns = [f"{c}_lag_{k}" for c in FEATURES]
#         feature_cols.append(shifted)
    
#     X_df = pd.concat(feature_cols, axis=1)
    
#     meta = pd.DataFrame({
#         "station": df["station_name"],
#         "hist_start": df["date"],
#         "future_start": df["date"].shift(-HIST_DAYS) 
#     })

#     # --- Construct Labels (y) & Valid Mask ---
#     y = None
#     if use_labels:
#         flood_future_start = df["flood"].shift(-HIST_DAYS)
#         indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=FUTURE_DAYS)
#         y = flood_future_start.rolling(window=indexer, min_periods=1).max()
#         valid_mask = meta["future_start"].notna() & y.notna()
#     else:
#         valid_mask = meta["future_start"].notna()

#     # Relaxed filtering: Allow rows with some NaNs (XGBoost handles them)
#     final_valid = valid_mask
    
#     X_final = X_df[final_valid].values
#     meta_final = meta[final_valid].copy()
    
#     if use_labels:
#         y_final = y[final_valid].fillna(0).astype(int).values
#         return X_final, y_final, meta_final
    
#     return X_final, None, meta_final

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--train_hourly", default="/home/georgechang/OOD/data/train_hourly.csv")
#     ap.add_argument("--test_hourly", default="/home/georgechang/OOD/data/test_hourly.csv")
#     ap.add_argument("--test_index", default="/home/georgechang/OOD/data/test_index.csv")
#     ap.add_argument("--predictions_out", default="/home/georgechang/OOD/output/predictions.csv")
#     args = ap.parse_args()

#     print("Loading data...")
#     train = pd.read_csv(args.train_hourly)
#     test  = pd.read_csv(args.test_hourly)
#     index = pd.read_csv(args.test_index)

#     print("Aggregating daily data (Robust)...")
#     daily_tr = daily_aggregate_vectorized(train)
#     daily_te = daily_aggregate_vectorized(test)

#     # Use Preloaded Thresholds
#     print("Loading preloaded thresholds...")
#     thr = get_preloaded_thresholds()
    
#     print(f"Train Data Construction...")
#     stn_tr = daily_tr["station_name"].unique().tolist()
#     X_tr, y_tr, _ = build_features_vectorized(daily_tr, stn_tr, use_labels=True, thresholds=thr)

#     print(f"Train Data Shape: {X_tr.shape}, Labels Shape: {y_tr.shape}")

#     pos = int((y_tr==1).sum())
#     neg = int((y_tr==0).sum())
#     spw = float(neg/max(pos, 1))
    
#     print(f"Training XGBoost (Pos: {pos}, Neg: {neg}, SPW: {spw:.2f})...")
#     clf = XGBClassifier(random_state=42, **({"n_estimators":600} if _HAS_XGB else {}))
    
#     if _HAS_XGB:
#         clf.set_params(
#             max_depth=5,
#             learning_rate=0.03,
#             subsample=0.8,
#             colsample_bytree=0.7,
#             reg_lambda=1.5,
#             reg_alpha=0.1,
#             objective="binary:logistic", 
#             eval_metric="auc",
#             n_jobs=-1, 
#             scale_pos_weight=spw
#         )
#     clf.fit(X_tr, y_tr)

#     print("Predicting...")
#     stn_te = daily_te["station_name"].unique().tolist()
#     X_te, _, meta_te = build_features_vectorized(daily_te, stn_te, use_labels=False)

#     if len(X_te) == 0:
#         raise RuntimeError("No test windows could be built. Check dates.")

#     probs = clf.predict_proba(X_te)[:,1] if hasattr(clf,"predict_proba") else clf.predict(X_te)
    
#     # Result Alignment
#     pred_df = meta_te.copy()
#     pred_df["y_prob"] = probs
    
#     index["hist_start"] = pd.to_datetime(index["hist_start"])
#     index["future_start"] = pd.to_datetime(index["future_start"])
    
#     out = index.merge(
#         pred_df, 
#         left_on=["station_name", "hist_start", "future_start"], 
#         right_on=["station", "hist_start", "future_start"], 
#         how="left"
#     )

#     result = out[["id", "y_prob"]].copy()
    
#     missing_count = result["y_prob"].isna().sum()
#     if missing_count > 0:
#         print(f"Warning: {missing_count} rows still not predicted (filling with 0.5).")
        
#     result["y_prob"] = result["y_prob"].fillna(0.5)
    
#     result.to_csv(args.predictions_out, index=False)
#     print(f"Wrote {args.predictions_out}")

# if __name__ == "__main__":
#     main()