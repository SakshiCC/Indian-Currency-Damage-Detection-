"""
evaluation.py
============================================================
Multi-label damage classification evaluation module.
Evaluates model predictions against human-verified annotations.
Calculates Micro/Macro F1, Hamming Loss, Jaccard Score, Exact Match Ratio,
and per-class Precision, Recall, and F1.

CLI Entry Point:
    python clip_damage_classifier/evaluation.py
============================================================
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple


def validate_human_annotations(
    verified_rows: List[Dict[str, str]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """
    Validates the integrity of human-annotated rows:
    1. Checks review_status == 'verified'
    2. Checks presence and valid binary format ('0' or '1') for all 5 categories.
    3. Detects incomplete/missing rows (evaluation must stop if any exist).
    4. Validates Normal constraint: Normal = 1 only when Torn=0, Folded=0, Burnt=0, Stain=0.
    """
    counts = {
        "total_rows": len(verified_rows),
        "verified_status": 0,
        "valid_torn": 0,
        "valid_folded": 0,
        "valid_burnt": 0,
        "valid_stain": 0,
        "valid_normal": 0
    }

    incomplete_rows = []
    normal_violations = []
    categories = ["Torn", "Folded", "Burnt", "Stain", "Normal"]

    for idx, r in enumerate(verified_rows, start=2):  # Line 2 corresponds to first data row
        rel = r.get("relative_path", "").strip()
        status = r.get("review_status", "").strip().lower()

        if status == "verified":
            counts["verified_status"] += 1

        row_missing_fields = []
        label_vals = {}
        for c in categories:
            col_name = f"human_{c}"
            raw_val = r.get(col_name)
            if raw_val is None or str(raw_val).strip() not in ("0", "1"):
                row_missing_fields.append(col_name)
            else:
                label_vals[c] = int(str(raw_val).strip())
                counts[f"valid_{c.lower()}"] += 1

        if row_missing_fields or status != "verified":
            incomplete_rows.append({
                "csv_line": idx,
                "relative_path": rel,
                "review_status": status,
                "missing_or_invalid": row_missing_fields
            })
        else:
            # Normal validation: Normal = 1 only when Torn=Folded=Burnt=Stain=0
            t = label_vals["Torn"]
            f = label_vals["Folded"]
            b = label_vals["Burnt"]
            s = label_vals["Stain"]
            n = label_vals["Normal"]
            if n == 1 and (t + f + b + s > 0):
                normal_violations.append({
                    "csv_line": idx,
                    "relative_path": rel,
                    "Torn": t,
                    "Folded": f,
                    "Burnt": b,
                    "Stain": s,
                    "Normal": n
                })

    return incomplete_rows, normal_violations, counts


def evaluate_predictions_against_verified(
    predictions_csv_path: str,
    verified_csv_path: str,
    report_txt_path: str = None,
    metrics_json_path: str = None,
    print_results: bool = True
) -> Dict[str, Any]:
    """
    Evaluates predictions against human-verified annotations.
    """
    p_path = Path(predictions_csv_path)
    v_path = Path(verified_csv_path)

    if not p_path.exists():
        msg = f"Predictions CSV not found: {p_path}"
        if print_results:
            print(f"[ERROR] {msg}")
        return {"status": "error", "message": msg}

    if not v_path.exists() or v_path.stat().st_size == 0:
        msg = f"Human verification CSV not found or empty: {v_path}"
        if print_results:
            print(f"[ERROR] {msg}")
        return {"status": "error", "message": msg}

    # Output paths
    if report_txt_path is None:
        report_txt_path = str(p_path.parent / "evaluation_report.txt")
    if metrics_json_path is None:
        metrics_json_path = str(p_path.parent / "evaluation_metrics.json")

    # Read verified CSV
    with open(v_path, "r", encoding="utf-8", newline="") as f:
        v_reader = csv.DictReader(f)
        raw_verified_rows = list(v_reader)

    # 1. Verification checks
    incomplete_rows, normal_violations, quality_counts = validate_human_annotations(raw_verified_rows)

    if incomplete_rows:
        error_msg = (
            f"ERROR: Found {len(incomplete_rows)} incomplete or unverified rows in {v_path.name}.\n"
            f"Missing labels must be provided before evaluation can proceed.\n"
        )
        for ir in incomplete_rows:
            error_msg += f"  - CSV Line {ir['csv_line']}: {ir['relative_path']} (status='{ir['review_status']}', missing={ir['missing_or_invalid']})\n"
        if print_results:
            print("============================================================")
            print("INCOMPLETE HUMAN VERIFICATION LABELS DETECTED")
            print("============================================================")
            print(error_msg)
        return {"status": "incomplete_annotations", "message": error_msg, "incomplete_rows": incomplete_rows}

    # Load verified dictionary keyed by relative_path
    verified_dict = {}
    for r in raw_verified_rows:
        rel = r.get("relative_path", "").strip()
        if rel and r.get("review_status", "").strip().lower() == "verified":
            verified_dict[rel] = {
                "Torn": int(r["human_Torn"].strip()),
                "Folded": int(r["human_Folded"].strip()),
                "Burnt": int(r["human_Burnt"].strip()),
                "Stain": int(r["human_Stain"].strip()),
                "Normal": int(r["human_Normal"].strip())
            }

    # Load predictions dictionary keyed by relative_path
    predictions_dict = {}
    with open(p_path, "r", encoding="utf-8", newline="") as f:
        p_reader = csv.DictReader(f)
        for r in p_reader:
            rel = r.get("relative_path", "").strip()
            if rel:
                predictions_dict[rel] = {
                    "Torn": int(r.get("Torn", 0)),
                    "Folded": int(r.get("Folded", 0)),
                    "Burnt": int(r.get("Burnt", 0)),
                    "Stain": int(r.get("Stain", 0)),
                    "Normal": int(r.get("Normal", 0))
                }

    # Match keys
    common_keys = sorted(list(set(verified_dict.keys()) & set(predictions_dict.keys())))
    if not common_keys:
        msg = f"No matching relative_path entries between predictions ({len(predictions_dict)}) and verified ({len(verified_dict)})."
        if print_results:
            print(f"[ERROR] {msg}")
        return {"status": "no_matches", "message": msg}

    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        hamming_loss,
        jaccard_score,
        accuracy_score
    )

    categories = ["Torn", "Folded", "Burnt", "Stain", "Normal"]
    y_true = [[verified_dict[k][c] for c in categories] for k in common_keys]
    y_pred = [[predictions_dict[k][c] for c in categories] for k in common_keys]

    # Calculate multi-label overall metrics
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    h_loss = hamming_loss(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred, average="samples", zero_division=0)
    exact_match = accuracy_score(y_true, y_pred)

    # Per-class metrics
    per_class = {}
    for idx, c in enumerate(categories):
        yt = [row[idx] for row in y_true]
        yp = [row[idx] for row in y_pred]
        p = precision_score(yt, yp, zero_division=0)
        r = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)
        support = sum(yt)
        pred_count = sum(yp)
        per_class[c] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "support": int(support),
            "predicted_count": int(pred_count)
        }

    # Construct report string
    report_lines = []
    report_lines.append("============================================================")
    report_lines.append("MULTI-LABEL DAMAGE CLASSIFICATION EVALUATION")
    report_lines.append("============================================================")
    report_lines.append("")
    report_lines.append(f"Verified sample size: {len(common_keys)}")
    report_lines.append(f"Matching prediction records: {len(common_keys)}")
    report_lines.append(f"Verified records: {len(verified_dict)}")
    report_lines.append("")
    report_lines.append("Verification Data Quality Checks")
    report_lines.append("------------------------------------------------------------")
    report_lines.append(f"review_status = verified: {quality_counts['verified_status']} / {quality_counts['total_rows']}")
    report_lines.append(f"Valid human_Torn:         {quality_counts['valid_torn']} / {quality_counts['total_rows']}")
    report_lines.append(f"Valid human_Folded:       {quality_counts['valid_folded']} / {quality_counts['total_rows']}")
    report_lines.append(f"Valid human_Burnt:        {quality_counts['valid_burnt']} / {quality_counts['total_rows']}")
    report_lines.append(f"Valid human_Stain:        {quality_counts['valid_stain']} / {quality_counts['total_rows']}")
    report_lines.append(f"Valid human_Normal:       {quality_counts['valid_normal']} / {quality_counts['total_rows']}")
    report_lines.append("")

    if normal_violations:
        report_lines.append("Normal Annotation Validation")
        report_lines.append("------------------------------------------------------------")
        report_lines.append(f"Notice: {len(normal_violations)} row(s) have human_Normal=1 while damage category=1.")
        report_lines.append("Rule: Normal = 1 only when Torn=0, Folded=0, Burnt=0, Stain=0.")
        report_lines.append("Problematic rows (preserved as annotated, not modified automatically):")
        for nv in normal_violations:
            report_lines.append(
                f"  - CSV Line {nv['csv_line']}: {nv['relative_path']} "
                f"(T={nv['Torn']}, F={nv['Folded']}, B={nv['Burnt']}, S={nv['Stain']}, N={nv['Normal']})"
            )
        report_lines.append("")
    else:
        report_lines.append("Normal Annotation Validation")
        report_lines.append("------------------------------------------------------------")
        report_lines.append("All annotations obey: Normal = 1 iff Torn = Folded = Burnt = Stain = 0.")
        report_lines.append("")

    report_lines.append("Overall Metrics")
    report_lines.append("------------------------------------------------------------")
    report_lines.append(f"Micro F1:              {micro_f1:.4f}")
    report_lines.append(f"Macro F1:              {macro_f1:.4f}")
    report_lines.append(f"Hamming Loss:          {h_loss:.4f}")
    report_lines.append(f"Jaccard Score:         {jaccard:.4f}")
    report_lines.append(f"Exact Match Ratio:     {exact_match:.4f}")
    report_lines.append("")
    report_lines.append("Per-Class Metrics")
    report_lines.append("------------------------------------------------------------")
    report_lines.append(f"{'Class':<12}{'Precision':<13}{'Recall':<10}{'F1':<10}")
    report_lines.append("-----------------------------------------")
    for c in categories:
        pc = per_class[c]
        report_lines.append(f"{c:<12}{pc['precision']:<13.4f}{pc['recall']:<10.4f}{pc['f1']:<10.4f}")
    report_lines.append("============================================================")

    report_text = "\n".join(report_lines)

    if print_results:
        print(report_text)

    # Save evaluation_report.txt
    try:
        Path(report_txt_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_txt_path, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")
        if print_results:
            print(f"\nSaved evaluation report to: {report_txt_path}")
    except Exception as e:
        if print_results:
            print(f"[WARNING] Could not save report text: {e}")

    # Save evaluation_metrics.json
    results_json = {
        "status": "success",
        "sample_size": len(common_keys),
        "matching_prediction_records": len(common_keys),
        "verified_records": len(verified_dict),
        "quality_counts": quality_counts,
        "normal_violations_count": len(normal_violations),
        "normal_violations": normal_violations,
        "overall_metrics": {
            "micro_f1": float(micro_f1),
            "macro_f1": float(macro_f1),
            "hamming_loss": float(h_loss),
            "jaccard_score": float(jaccard),
            "exact_match_ratio": float(exact_match)
        },
        "per_class_metrics": per_class
    }

    try:
        Path(metrics_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(results_json, f, indent=2)
        if print_results:
            print(f"Saved evaluation metrics JSON to: {metrics_json_path}")
    except Exception as e:
        if print_results:
            print(f"[WARNING] Could not save metrics JSON: {e}")

    return results_json


def main():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate multi-label banknote damage predictions against human verification sample.")
    parser.add_argument("--version", choices=["v1", "v2", "v3"], default="v1", help="Evaluation version (v1 baseline, v2 calibrated, or v3 dynamic)")
    parser.add_argument("--predictions", "-p", type=str, default=None, help="Path to predictions CSV")
    parser.add_argument("--verified", "-v", type=str, default=None, help="Path to human verification CSV")
    parser.add_argument("--report", "-r", type=str, default=None, help="Path to save evaluation report txt")
    parser.add_argument("--json", "-j", type=str, default=None, help="Path to save evaluation metrics JSON")

    args = parser.parse_args()

    # Determine default paths based on version
    if args.version == "v3":
        default_pred = base_dir / "outputs" / "v3" / "spoilt_indian_banknotes_damage_classification_v3.csv"
        default_report = base_dir / "outputs" / "v3" / "evaluation_report_v3.txt"
        default_json = base_dir / "outputs" / "v3" / "evaluation_metrics_v3.json"
    elif args.version == "v2":
        default_pred = base_dir / "outputs" / "spoilt_indian_banknotes_damage_classification_v2.csv"
        default_report = base_dir / "outputs" / "evaluation_report_v2.txt"
        default_json = base_dir / "outputs" / "evaluation_metrics_v2.json"
    else:
        default_pred = base_dir / "outputs" / "spoilt_indian_banknotes_damage_classification.csv"
        default_report = base_dir / "outputs" / "evaluation_report.txt"
        default_json = base_dir / "outputs" / "evaluation_metrics.json"

    pred_path = args.predictions or str(default_pred)
    ver_path = args.verified or str(base_dir / "outputs" / "human_verification_sample.csv")
    report_path = args.report or str(default_report)
    json_path = args.json or str(default_json)

    results = evaluate_predictions_against_verified(
        predictions_csv_path=pred_path,
        verified_csv_path=ver_path,
        report_txt_path=report_path,
        metrics_json_path=json_path,
        print_results=True
    )

    if results.get("status") not in ("success",):
        sys.exit(1)



if __name__ == "__main__":
    main()
