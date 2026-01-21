#!/usr/bin/env python3
"""
Scoring program: binary classification
Supports BOTH invocation styles:
1) Positional (classic CodaLab):
   python3 scoring.py <solution_dir> <prediction_dir> <score_dir>

2) Flags:
   python3 -u scoring.py --solution_dir $input/ref --prediction_dir $input/res --score_dir $output
"""
import argparse, json, pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
import sys

def read_gt(solution_dir: Path) -> pd.DataFrame:
    print(solution_dir)
    gt = pd.read_csv(str(solution_dir)+"/y_test.csv")
    if "id" not in gt.columns: gt.insert(0,"id",range(len(gt)))
    if "y_true" not in gt.columns:
        last = [c for c in gt.columns if c!="id"][-1]
        gt = gt.rename(columns={last:"y_true"})
    return gt[["id","y_true"]]

def read_preds(pred_dir: Path) -> pd.DataFrame:
    p = str(pred_dir)+"/predictions.csv"
    p1 = Path(p)
    if not p1.exists(): raise FileNotFoundError("predictions.csv not found")
    pred = pd.read_csv(p)
    if "id" not in pred.columns: pred.insert(0,"id",range(len(pred)))
    return pred

def score(solution_dir: Path, prediction_dir: Path, score_dir: Path):
    score_dir.mkdir(parents=True, exist_ok=True)
    gt = read_gt(solution_dir); pr = read_preds(prediction_dir)
    df = gt.merge(pr, on="id", how="inner")
    if len(df)==0: raise ValueError("No overlapping ids between y_test and predictions.")

    y_true = df["y_true"].astype(int).to_numpy()
    if "y_prob" in df.columns:
        y_prob = df["y_prob"].astype(float).to_numpy()
    elif "label" in df.columns:
        y_prob = df["label"].astype(int).to_numpy()
    else:
        y_prob = df.iloc[:, -1].astype(float).to_numpy()

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    y_pred = (y_prob >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    out = {
        "auc": auc,
        "acc": acc,
        "f1": f1,
        "mcc": mcc,
        "n": int(len(y_true))  # optional extra field
    }
    (score_dir/"scores.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

def main():
    # POSitional
    # if len(sys.argv) >= 4 and not sys.argv[1].startswith("--"):
    #     solution_dir = Path(sys.argv[1])
    #     prediction_dir = Path(sys.argv[2])
    #     score_dir = Path(sys.argv[3])
    #     score(solution_dir, prediction_dir, score_dir)
    #     return


    # FLAGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution_dir", required=True)
    ap.add_argument("--prediction_dir", required=True)
    ap.add_argument("--score_dir", required=True)
    args = ap.parse_args()

    solution_dir = Path(args.solution_dir)
    prediction_dir = Path(args.prediction_dir)
    score_dir = Path(args.score_dir)
    
    score(solution_dir, prediction_dir, score_dir)

if __name__ == "__main__":
    main()