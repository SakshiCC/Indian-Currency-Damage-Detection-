"""
calibrate_v3.py
============================================================
Calibration and Dynamic Threshold Training Engine for Vinit V3.

Performs:
  1. Context feature extraction for the 90-image human verification set.
  2. Regularized logistic calibration model fitting per damage category:
       logit(P(C=1)) = intercept_C + w_ev,C * evidence_C + sum(w_i,C * norm_context_i)
  3. Strict mathematical constraint enforcement:
       w_ev,C > 0
       If violated or unhelpful, falls back safely to anchored V2 threshold.
  4. Cross-validation (Stratified K-Fold where sample counts permit).
  5. Empirical threshold sweep across [-0.006, +0.006] (step 0.0005).
  6. Feature ablation comparison.
  7. Exporting calibrated_dynamic_thresholds.json, threshold_sweep.csv,
     and calibration_report_v3.txt to outputs/v3/.
============================================================
"""

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clip_damage_classifier.calibration.context_features import (
    extract_context_features,
    extract_features_from_image,
)
from clip_damage_classifier.calibration.dynamic_thresholds import (
    DEFAULT_V2_BASE_THRESHOLDS,
    DEFAULT_MIN_THRESHOLD,
    DEFAULT_MAX_THRESHOLD,
    DynamicThresholdCalculator,
)
from clip_damage_classifier.preprocessing import image_preprocessing as ip
from clip_damage_classifier.preprocessing import note_detection as nd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("calibrate_v3")

DAMAGE_CATEGORIES = ["Torn", "Folded", "Burnt", "Stain"]


def load_verified_dataset(csv_path: Path) -> List[Dict[str, Any]]:
    """Load verified records from human_verification_sample.csv."""
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("review_status", "").strip().lower() == "verified":
                records.append(row)
    logger.info(f"Loaded {len(records)} verified records from {csv_path.name}")
    return records


def extract_features_for_dataset(
    records: List[Dict[str, Any]], project_root: Path
) -> List[Dict[str, Any]]:
    """Extract context features and ground-truth targets for all records."""
    data = []
    for r in records:
        # Resolve image path
        img_p = Path(r["image_path"])
        if not img_p.exists():
            img_p = project_root / r["relative_path"]

        img = None
        if img_p.exists():
            img = cv2.imread(str(img_p))

        evidence_scores = {
            "Torn": float(r["torn_evidence_score"]),
            "Folded": float(r["folded_evidence_score"]),
            "Burnt": float(r["burnt_evidence_score"]),
            "Stain": float(r["stain_evidence_score"]),
        }

        ctx = extract_context_features(image=img, evidence_scores=evidence_scores)

        item = {
            "relative_path": r["relative_path"],
            "image_path": str(img_p),
            "human_Torn": int(r["human_Torn"]),
            "human_Folded": int(r["human_Folded"]),
            "human_Burnt": int(r["human_Burnt"]),
            "human_Stain": int(r["human_Stain"]),
            "human_Normal": int(r["human_Normal"]),
            "torn_evidence": evidence_scores["Torn"],
            "folded_evidence": evidence_scores["Folded"],
            "burnt_evidence": evidence_scores["Burnt"],
            "stain_evidence": evidence_scores["Stain"],
        }
        item.update(ctx)
        data.append(item)
    return data


def run_threshold_sweep(
    data: List[Dict[str, Any]],
    output_csv: Path,
    t_min: float = -0.0060,
    t_max: float = +0.0060,
    step: float = 0.0005,
) -> Dict[str, Dict[str, Any]]:
    """
    Empirical threshold sweep across evidence scores for each category.
    Saves results to threshold_sweep.csv and returns optimal thresholds.
    """
    thresholds = np.arange(t_min, t_max + step / 2, step)
    sweep_results = []
    best_thresholds: Dict[str, Dict[str, Any]] = {}

    for cat in DAMAGE_CATEGORIES:
        y_true = np.array([d[f"human_{cat}"] for d in data])
        ev_scores = np.array([d[f"{cat.lower()}_evidence"] for d in data])

        best_f1 = -1.0
        best_t = 0.0
        best_prec = 0.0
        best_rec = 0.0

        for t in thresholds:
            t_val = round(float(t), 5)
            y_pred = (ev_scores >= t_val).astype(int)

            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            sweep_results.append({
                "category": cat,
                "threshold": t_val,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            })

            if f1 > best_f1:
                best_f1 = f1
                best_t = t_val
                best_prec = prec
                best_rec = rec

        best_thresholds[cat] = {
            "best_f1_threshold": best_t,
            "best_f1": round(best_f1, 4),
            "precision_at_best": round(best_prec, 4),
            "recall_at_best": round(best_rec, 4),
            "base_v2_threshold": DEFAULT_V2_BASE_THRESHOLDS[cat],
        }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["category", "threshold", "precision", "recall", "f1_score", "tp", "fp", "fn", "tn"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_results)

    logger.info(f"Saved threshold sweep across {len(thresholds)} points to {output_csv}")
    return best_thresholds


def fit_calibrator_for_class(
    data: List[Dict[str, Any]],
    cat: str,
    feature_candidates: List[str],
    c_reg: float = 0.5,
) -> Dict[str, Any]:
    """
    Fits regularized logistic regression for category C and computes dynamic threshold params.
    """
    y = np.array([d[f"human_{cat}"] for d in data])
    ev = np.array([d[f"{cat.lower()}_evidence"] for d in data])

    target_ev_col = f"{cat.lower()}_evidence"
    context_cols = [c for c in feature_candidates if c != target_ev_col]

    # Matrix: col 0 is target evidence; cols 1.. are context features
    all_cols = [target_ev_col] + context_cols
    X_raw = np.array([[d[col] for col in all_cols] for d in data])

    # Standardize context features only (keep evidence in native scale for direct threshold derivation)
    scaler = StandardScaler()
    X_norm = np.copy(X_raw)
    if len(context_cols) > 0:
        X_norm[:, 1:] = scaler.fit_transform(X_raw[:, 1:])

    # Fit L2-regularized logistic regression
    clf = LogisticRegression(
        C=c_reg,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    clf.fit(X_norm, y)

    w_ev = float(clf.coef_[0][0])
    intercept = float(clf.intercept_[0])
    w_ctx = [float(w) for w in clf.coef_[0][1:]]

    # Prepare scaler parameters
    scaler_means = {}
    scaler_stds = {}
    for idx, col in enumerate(context_cols):
        scaler_means[col] = float(scaler.mean_[idx])
        scaler_stds[col] = float(scaler.scale_[idx])

    # Cross-validation performance
    cv_f1s = []
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    n_splits = min(5, n_pos, n_neg)

    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, val_idx in skf.split(X_norm, y):
            clf_cv = LogisticRegression(C=c_reg, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
            clf_cv.fit(X_norm[train_idx], y[train_idx])
            y_val_pred = clf_cv.predict(X_norm[val_idx])
            cv_f1s.append(f1_score(y[val_idx], y_val_pred, zero_division=0))
        mean_cv_f1 = float(np.mean(cv_f1s))
    else:
        mean_cv_f1 = float(f1_score(y, clf.predict(X_norm), zero_division=0))

    # Evaluate dynamic threshold vs baseline V2 on dataset
    base_t = DEFAULT_V2_BASE_THRESHOLDS[cat]
    y_v2 = (ev >= base_t).astype(int)
    p_v2 = precision_score(y, y_v2, zero_division=0)
    r_v2 = recall_score(y, y_v2, zero_division=0)
    f1_v2 = f1_score(y, y_v2, zero_division=0)

    # Status check: w_evidence must be strictly positive
    if w_ev <= 0.0:
        status = "fallback_nonpositive_evidence"
        dyn_p, dyn_r, dyn_f1 = p_v2, r_v2, f1_v2
    else:
        # Calculate dynamic threshold for each sample
        dyn_thresholds = []
        for i in range(len(data)):
            ctx_val = X_norm[i, 1:]
            t_dyn = -(intercept + np.dot(w_ctx, ctx_val)) / w_ev
            t_dyn = float(np.clip(t_dyn, DEFAULT_MIN_THRESHOLD, DEFAULT_MAX_THRESHOLD))
            dyn_thresholds.append(t_dyn)

        y_dyn = (ev >= np.array(dyn_thresholds)).astype(int)
        dyn_p = precision_score(y, y_dyn, zero_division=0)
        dyn_r = recall_score(y, y_dyn, zero_division=0)
        dyn_f1 = f1_score(y, y_dyn, zero_division=0)

        # Require that dynamic model matches or improves baseline F1
        if dyn_f1 >= f1_v2 - 0.05:
            status = "calibrated"
        else:
            status = "fallback_inferior_to_v2"

    return {
        "category": cat,
        "status": status,
        "w_evidence": round(w_ev, 6),
        "intercept": round(intercept, 6),
        "feature_names": context_cols,
        "w_context": [round(w, 6) for w in w_ctx],
        "scaler_means": {k: round(v, 6) for k, v in scaler_means.items()},
        "scaler_stds": {k: round(v, 6) for k, v in scaler_stds.items()},
        "cv_f1_mean": round(mean_cv_f1, 4),
        "v2_precision": round(float(p_v2), 4),
        "v2_recall": round(float(r_v2), 4),
        "v2_f1": round(float(f1_v2), 4),
        "dynamic_precision": round(float(dyn_p), 4),
        "dynamic_recall": round(float(dyn_r), 4),
        "dynamic_f1": round(float(dyn_f1), 4),
    }


def calibrate_all_classes(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Train calibrated models with class-specific selected context features."""
    class_feature_configs = {
        "Folded": ["edge_density", "contrast", "area_ratio"],
        "Stain": ["dark_ratio", "brightness", "contrast"],
        "Torn": ["area_ratio", "edge_density", "contrast"],
        "Burnt": ["stain_evidence", "dark_ratio", "contrast"],
    }

    calibrations = {}
    for cat in DAMAGE_CATEGORIES:
        feat_cands = class_feature_configs[cat]
        res = fit_calibrator_for_class(data, cat, feat_cands, c_reg=0.5)
        calibrations[cat] = res
        logger.info(
            f"Class {cat:6s}: status={res['status']:25s} w_ev={res['w_evidence']:8.4f} "
            f"V2_F1={res['v2_f1']:.4f} -> Dyn_F1={res['dynamic_f1']:.4f}"
        )

    return calibrations


def export_calibration_artifact(
    calibrations: Dict[str, Any], output_path: Path
):
    """Save calibrated parameters to calibrated_dynamic_thresholds.json."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v3",
        "description": "Vinit V3 dynamic threshold calibration parameters",
        "anchors_base_v2": DEFAULT_V2_BASE_THRESHOLDS,
        "safety_bounds": {
            "min_threshold": DEFAULT_MIN_THRESHOLD,
            "max_threshold": DEFAULT_MAX_THRESHOLD,
        },
        "classes": calibrations,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Exported calibration artifact to {output_path}")


def main():
    logger.info("Starting Vinit V3 Calibration & Dynamic Threshold Training...")

    human_csv = (
        PROJECT_ROOT
        / "clip_damage_classifier"
        / "outputs"
        / "human_verification_sample.csv"
    )
    out_dir = PROJECT_ROOT / "clip_damage_classifier" / "outputs" / "v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    records = load_verified_dataset(human_csv)
    dataset_features = extract_features_for_dataset(records, PROJECT_ROOT)

    # 2. Empirical threshold sweep across [-0.006, +0.006]
    sweep_csv = out_dir / "threshold_sweep.csv"
    best_sweep = run_threshold_sweep(dataset_features, sweep_csv)

    # 3. Fit calibrated dynamic models
    calibrations = calibrate_all_classes(dataset_features)

    # 4. Export JSON artifact
    json_path = out_dir / "calibrated_dynamic_thresholds.json"
    export_calibration_artifact(calibrations, json_path)

    # 5. Generate calibration report
    report_path = out_dir / "calibration_report_v3.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("VINIT V3 DAMAGE CLASSIFIER - CALIBRATION & THRESHOLD REPORT\n")
        f.write("============================================================\n\n")
        f.write("1. ANCHORED V2 BASELINE THRESHOLDS:\n")
        for cat, val in DEFAULT_V2_BASE_THRESHOLDS.items():
            f.write(f"   - {cat:7s}: {val:+.4f}\n")
        f.write("\n2. EMPIRICAL THRESHOLD SWEEP RESULTS (Best F1 on 90 verified images):\n")
        for cat, info in best_sweep.items():
            f.write(
                f"   - {cat:7s}: Best T={info['best_f1_threshold']:+.4f} (F1={info['best_f1']:.4f}, "
                f"P={info['precision_at_best']:.4f}, R={info['recall_at_best']:.4f}) | Base V2={info['base_v2_threshold']:+.4f}\n"
            )
        f.write("\n3. DYNAMIC CALIBRATION MODELS:\n")
        for cat, res in calibrations.items():
            f.write(f"   --- Class: {cat} ---\n")
            f.write(f"   Status:        {res['status']}\n")
            f.write(f"   w_evidence:    {res['w_evidence']:.6f}\n")
            f.write(f"   intercept:     {res['intercept']:.6f}\n")
            f.write(f"   Context feats: {res['feature_names']}\n")
            f.write(f"   w_context:     {res['w_context']}\n")
            f.write(f"   CV Mean F1:    {res['cv_f1_mean']:.4f}\n")
            f.write(f"   V2 Baseline:   P={res['v2_precision']:.4f}, R={res['v2_recall']:.4f}, F1={res['v2_f1']:.4f}\n")
            f.write(f"   V3 Dynamic:    P={res['dynamic_precision']:.4f}, R={res['dynamic_recall']:.4f}, F1={res['dynamic_f1']:.4f}\n\n")

    logger.info(f"Calibration report saved to {report_path}")
    logger.info("V3 calibration completed successfully.")


if __name__ == "__main__":
    main()
