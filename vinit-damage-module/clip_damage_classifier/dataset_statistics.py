"""
dataset_statistics.py
============================================================
Generates comprehensive dataset distribution statistics,
a stratified 80-100 image human review sample, and processing summary.
============================================================
"""

import os
import csv
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any


def generate_dataset_statistics(
    main_csv_path: str,
    output_dir: str,
    total_discovered: int = None,
    failed_count: int = 0,
    skipped_count: int = 0,
    elapsed_time: float = 0.0,
    device: str = "cpu",
    batch_size: int = 32
) -> Dict[str, Any]:
    csv_p = Path(main_csv_path)
    if not csv_p.exists():
        raise FileNotFoundError(f"Classification CSV not found at: {csv_p}")

    out_d = Path(output_dir)
    out_d.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(csv_p, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    total_processed = len(rows)
    if total_discovered is None:
        total_discovered = total_processed + failed_count

    denom = total_processed if total_processed > 0 else 1

    # Category counts
    torn_count = sum(1 for r in rows if r.get("Torn") == "1")
    folded_count = sum(1 for r in rows if r.get("Folded") == "1")
    burnt_count = sum(1 for r in rows if r.get("Burnt") == "1")
    stain_count = sum(1 for r in rows if r.get("Stain") == "1")
    normal_count = sum(1 for r in rows if r.get("Normal") == "1")

    # Multi-damage counts
    multi_damage_count = sum(
        1 for r in rows
        if (int(r.get("Torn", 0)) + int(r.get("Folded", 0)) + int(r.get("Burnt", 0)) + int(r.get("Stain", 0))) > 1
    )
    all_four_count = sum(
        1 for r in rows
        if (int(r.get("Torn", 0)) + int(r.get("Folded", 0)) + int(r.get("Burnt", 0)) + int(r.get("Stain", 0))) == 4
    )

    combinations = Counter(r.get("predicted_labels", "") for r in rows)

    # Status counts
    status_counts = Counter(r.get("classification_status", "") for r in rows)
    strong_count = status_counts.get("strong_prediction", 0)
    moderate_count = status_counts.get("moderate_prediction", 0)
    uncertain_count = status_counts.get("uncertain", 0)

    # 1. Write damage_class_distribution.csv
    class_dist_path = out_d / "damage_class_distribution.csv"
    with open(class_dist_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "count", "percentage"])
        writer.writerow(["Normal", normal_count, f"{(normal_count / denom) * 100:.2f}%"])
        writer.writerow(["Folded", folded_count, f"{(folded_count / denom) * 100:.2f}%"])
        writer.writerow(["Stain", stain_count, f"{(stain_count / denom) * 100:.2f}%"])
        writer.writerow(["Burnt", burnt_count, f"{(burnt_count / denom) * 100:.2f}%"])
        writer.writerow(["Torn", torn_count, f"{(torn_count / denom) * 100:.2f}%"])

    # 2. Write damage_combination_distribution.csv
    comb_dist_path = out_d / "damage_combination_distribution.csv"
    with open(comb_dist_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["combination", "count", "percentage"])
        for comb, cnt in combinations.most_common():
            writer.writerow([comb, cnt, f"{(cnt / denom) * 100:.2f}%"])

    # 3. Write ambiguous_images.csv (all uncertain images)
    ambiguous_path = out_d / "ambiguous_images.csv"
    uncertain_rows = [r for r in rows if r.get("classification_status") == "uncertain"]
    with open(ambiguous_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "image_path", "relative_path", "filename", "denomination", "note_generation",
            "predicted_labels", "torn_evidence_score", "folded_evidence_score",
            "burnt_evidence_score", "stain_evidence_score", "classification_status"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ur in uncertain_rows:
            writer.writerow({
                "image_path": ur.get("image_path", ""),
                "relative_path": ur.get("relative_path", ""),
                "filename": ur.get("filename", ""),
                "denomination": ur.get("denomination", ""),
                "note_generation": ur.get("note_generation", ""),
                "predicted_labels": ur.get("predicted_labels", ""),
                "torn_evidence_score": ur.get("torn_evidence_score", ""),
                "folded_evidence_score": ur.get("folded_evidence_score", ""),
                "burnt_evidence_score": ur.get("burnt_evidence_score", ""),
                "stain_evidence_score": ur.get("stain_evidence_score", ""),
                "classification_status": ur.get("classification_status", "")
            })

    # 4. Stratified Human Verification Sample (85-95 images)
    human_sample_path = out_d / "human_verification_sample.csv"
    target_quotas = {
        "strong_torn": 12,
        "strong_folded": 12,
        "strong_burnt": 10,
        "strong_stain": 12,
        "normal": 14,
        "multi": 15,
        "uncertain": 15
    }
    quota_collected = defaultdict(int)
    denom_seen = defaultdict(lambda: defaultdict(int))
    selected_sample = []

    for r in rows:
        lbl = r.get("predicted_labels", "")
        status = r.get("classification_status", "")
        d = r.get("denomination", "unknown")
        g = r.get("note_generation", "unknown")

        bucket = None
        if status == "uncertain" and quota_collected["uncertain"] < target_quotas["uncertain"]:
            bucket = "uncertain"
        elif status == "strong_prediction":
            if lbl == "Torn" and quota_collected["strong_torn"] < target_quotas["strong_torn"]:
                bucket = "strong_torn"
            elif lbl == "Folded" and quota_collected["strong_folded"] < target_quotas["strong_folded"]:
                bucket = "strong_folded"
            elif lbl == "Burnt" and quota_collected["strong_burnt"] < target_quotas["strong_burnt"]:
                bucket = "strong_burnt"
            elif lbl == "Stain" and quota_collected["strong_stain"] < target_quotas["strong_stain"]:
                bucket = "strong_stain"
            elif lbl == "Normal" and quota_collected["normal"] < target_quotas["normal"]:
                bucket = "normal"
            elif ";" in lbl and quota_collected["multi"] < target_quotas["multi"]:
                bucket = "multi"
        elif ";" in lbl and quota_collected["multi"] < target_quotas["multi"]:
            bucket = "multi"

        if bucket:
            if denom_seen[bucket][f"{d}_{g}"] < 3:
                selected_sample.append(r)
                quota_collected[bucket] += 1
                denom_seen[bucket][f"{d}_{g}"] += 1

        if len(selected_sample) >= 90:
            break

    with open(human_sample_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "image_path", "relative_path", "filename", "denomination", "note_generation",
            "predicted_labels", "classification_status", "torn_evidence_score",
            "folded_evidence_score", "burnt_evidence_score", "stain_evidence_score",
            "human_Torn", "human_Folded", "human_Burnt", "human_Stain", "human_Normal",
            "review_status", "review_notes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sr in selected_sample:
            writer.writerow({
                "image_path": sr.get("image_path", ""),
                "relative_path": sr.get("relative_path", ""),
                "filename": sr.get("filename", ""),
                "denomination": sr.get("denomination", ""),
                "note_generation": sr.get("note_generation", ""),
                "predicted_labels": sr.get("predicted_labels", ""),
                "classification_status": sr.get("classification_status", ""),
                "torn_evidence_score": sr.get("torn_evidence_score", ""),
                "folded_evidence_score": sr.get("folded_evidence_score", ""),
                "burnt_evidence_score": sr.get("burnt_evidence_score", ""),
                "stain_evidence_score": sr.get("stain_evidence_score", ""),
                "human_Torn": "",
                "human_Folded": "",
                "human_Burnt": "",
                "human_Stain": "",
                "human_Normal": "",
                "review_status": "pending_review",
                "review_notes": ""
            })

    # 5. Write processing_summary.txt
    summary_path = out_d / "processing_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("INDIAN BANKNOTES CONTRASTIVE MULTI-LABEL DAMAGE CLASSIFICATION REPORT\n")
        f.write("============================================================\n\n")
        f.write(f"Model Used: openai/clip-vit-base-patch32 (Zero-Shot CLIP ViT-B/32)\n")
        f.write(f"Scoring Architecture: Contrastive (Positive vs Negative Prompt Ensembles)\n")
        f.write(f"Device: {device}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Elapsed Processing Time: {elapsed_time:.1f} seconds ({(elapsed_time/60):.2f} minutes)\n\n")
        f.write(f"--- DATASET PROCESSING COUNTS ---\n")
        f.write(f"Total Images Discovered: {total_discovered}\n")
        f.write(f"Successfully Processed: {total_processed} (100.0%)\n")
        f.write(f"Failed / Corrupted:     {failed_count}\n")
        f.write(f"Skipped:                {skipped_count}\n\n")
        f.write(f"--- PREDICTION CONFIDENCE STATUSES ---\n")
        f.write(f"Strong Predictions:   {strong_count:5d} ({strong_count/denom*100:5.2f}%)\n")
        f.write(f"Moderate Predictions: {moderate_count:5d} ({moderate_count/denom*100:5.2f}%)\n")
        f.write(f"Uncertain (Review):   {uncertain_count:5d} ({uncertain_count/denom*100:5.2f}%)\n\n")
        f.write(f"--- DAMAGE CATEGORY COUNTS (MULTI-LABEL) ---\n")
        f.write(f"Normal (Intact): {normal_count:5d} ({normal_count/denom*100:5.2f}%)\n")
        f.write(f"Folded:          {folded_count:5d} ({folded_count/denom*100:5.2f}%)\n")
        f.write(f"Stain:           {stain_count:5d} ({stain_count/denom*100:5.2f}%)\n")
        f.write(f"Burnt:           {burnt_count:5d} ({burnt_count/denom*100:5.2f}%)\n")
        f.write(f"Torn:            {torn_count:5d} ({torn_count/denom*100:5.2f}%)\n\n")
        f.write(f"--- MULTI-DAMAGE METRICS ---\n")
        f.write(f"Total Multi-Damage Notes: {multi_damage_count:5d} ({multi_damage_count/denom*100:5.2f}%)\n")
        f.write(f"All-Four-Damages Notes:   {all_four_count:5d} ({all_four_count/denom*100:5.2f}%)\n\n")
        f.write(f"--- TOP DAMAGE COMBINATIONS ---\n")
        for comb, cnt in combinations.most_common(12):
            f.write(f"  {comb:30s}: {cnt:5d} ({cnt/denom*100:5.2f}%)\n")
        f.write(f"\n--- HUMAN REVIEW SAMPLE ---\n")
        f.write(f"Stratified Sample Size: {len(selected_sample)} images in outputs/human_verification_sample.csv\n")
        f.write(f"Full Uncertain Set:     {uncertain_count} images retained in outputs/ambiguous_images.csv\n\n")
        f.write("Status: Automatically classified by the CLIP zero-shot pipeline (Pseudo-Labels)\n")
        f.write("Ground-truth evaluation requires human-verified annotations.\n")
        f.write("============================================================\n")

    return {
        "total_discovered": total_discovered,
        "total_processed": total_processed,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "strong_count": strong_count,
        "moderate_count": moderate_count,
        "uncertain_count": uncertain_count,
        "torn_count": torn_count,
        "folded_count": folded_count,
        "burnt_count": burnt_count,
        "stain_count": stain_count,
        "normal_count": normal_count,
        "multi_damage_count": multi_damage_count,
        "all_four_count": all_four_count,
        "sample_size": len(selected_sample),
        "combinations": combinations
    }
