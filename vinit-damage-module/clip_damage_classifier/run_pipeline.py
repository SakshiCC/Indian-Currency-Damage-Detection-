"""
run_pipeline.py
============================================================
Main CLI runner for Contrastive CLIP Multi-Label Damage Classification.
============================================================
"""

import argparse
import time
import sys
from pathlib import Path
from . import config
from .zip_extractor import safe_extract_zip
from .dataset_scanner import scan_images
from .resume_manager import get_already_processed_keys
from .csv_writer import IncrementalCSVWriter, ErrorLogWriter, AmbiguousCSVWriter
from .classifier import DamageClassifier
from .dataset_statistics import generate_dataset_statistics


def parse_args():
    parser = argparse.ArgumentParser(description="Contrastive CLIP Multi-Label Damage Classification")
    parser.add_argument("--zip", type=str, default=str(config.DEFAULT_ZIP_PATH))
    parser.add_argument("--extract-dir", type=str, default=str(config.DEFAULT_EXTRACT_DIR))
    parser.add_argument("--output-dir", type=str, default=str(config.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--device", type=str, default=config.DEVICE_PREFERENCE)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--multi-view", action="store_true", default=False)
    # Threshold overrides
    parser.add_argument("--torn-thresh", type=float, default=config.INITIAL_EVIDENCE_THRESHOLDS["Torn"])
    parser.add_argument("--folded-thresh", type=float, default=config.INITIAL_EVIDENCE_THRESHOLDS["Folded"])
    parser.add_argument("--burnt-thresh", type=float, default=config.INITIAL_EVIDENCE_THRESHOLDS["Burnt"])
    parser.add_argument("--stain-thresh", type=float, default=config.INITIAL_EVIDENCE_THRESHOLDS["Stain"])
    return parser.parse_args()


def run():
    args = parse_args()
    start_time = time.time()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    main_csv_path = output_dir / "spoilt_indian_banknotes_damage_classification.csv"
    error_csv_path = output_dir / "processing_errors.csv"
    ambiguous_csv_path = output_dir / "ambiguous_images.csv"

    # Determine overwrite vs resume
    do_overwrite = args.overwrite or (not args.resume)

    thresholds = {
        "Torn": args.torn_thresh,
        "Folded": args.folded_thresh,
        "Burnt": args.burnt_thresh,
        "Stain": args.stain_thresh,
    }

    print("============================================================")
    print("Contrastive CLIP Multi-Label Damage Classification Pipeline")
    print("============================================================")
    print(f"ZIP Path:         {args.zip}")
    print(f"Extract Dir:      {args.extract_dir}")
    print(f"Output Dir:       {args.output_dir}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Device:           {args.device}")
    print(f"Multi-View:       {args.multi_view}")
    print(f"Evidence Thresh:  {thresholds}")
    print(f"Limit:            {args.limit if args.limit is not None else 'ALL'}")
    print(f"Overwrite:        {do_overwrite}")
    print(f"Resume:           {not do_overwrite}")
    print("============================================================")

    # 1. Extract ZIP
    extracted_dir, _ = safe_extract_zip(args.zip, args.extract_dir)

    # 2. Scan dataset recursively
    all_images = scan_images(extracted_dir)
    total_discovered = len(all_images)
    print(f"[SCANNER] Discovered {total_discovered} valid images in dataset.")

    if total_discovered == 0:
        print("[ERROR] No valid images found. Exiting.")
        sys.exit(1)

    if args.limit is not None and args.limit > 0:
        all_images = all_images[:args.limit]
        print(f"[TEST MODE] Limiting run to first {len(all_images)} images.")

    # 3. Check resume status
    already_processed = set()
    if not do_overwrite:
        already_processed = get_already_processed_keys(str(main_csv_path))
        print(f"[RESUME] Found {len(already_processed)} previously processed images.")

    images_to_process = [img for img in all_images if img["relative_path"] not in already_processed]
    skipped_count = len(all_images) - len(images_to_process)
    print(f"[PIPELINE] Images to process: {len(images_to_process)} (Skipped: {skipped_count})")

    if not images_to_process:
        print("[PIPELINE] All images already processed! Generating statistics...")
        stats = generate_dataset_statistics(
            str(main_csv_path),
            str(output_dir),
            total_discovered=total_discovered,
            failed_count=0,
            skipped_count=skipped_count,
            elapsed_time=0.0,
            device=args.device,
            batch_size=args.batch_size
        )
        return

    # 4. Initialize CSV Writers
    csv_writer = IncrementalCSVWriter(str(main_csv_path), overwrite=do_overwrite)
    error_writer = ErrorLogWriter(str(error_csv_path), overwrite=do_overwrite)
    ambiguous_writer = AmbiguousCSVWriter(str(ambiguous_csv_path), overwrite=do_overwrite)

    # 5. Initialize Classifier
    classifier = DamageClassifier(
        thresholds=thresholds,
        device=args.device,
        multi_view=args.multi_view
    )

    # 6. Batch Processing Loop
    total_to_run = len(images_to_process)
    processed_this_run = 0
    failed_this_run = 0
    batch_size = args.batch_size

    print(f"[PIPELINE] Beginning batch processing of {total_to_run} images...")

    for i in range(0, total_to_run, batch_size):
        b_items = images_to_process[i: i + batch_size]
        b_start = time.time()

        results, failed = classifier.process_batch(b_items)

        if results:
            csv_writer.write_rows(results)
            uncertain_rows = [r for r in results if r.get("classification_status") == "uncertain"]
            if uncertain_rows:
                ambiguous_writer.write_rows(uncertain_rows)

        if failed:
            for fail_item in failed:
                error_writer.log_error(
                    fail_item["image_path"],
                    fail_item["relative_path"],
                    fail_item["error_type"],
                    fail_item["error_message"]
                )
            failed_this_run += len(failed)

        processed_this_run += len(results)
        b_elapsed = time.time() - b_start
        fps = len(b_items) / b_elapsed if b_elapsed > 0 else 0
        total_done = processed_this_run + skipped_count
        progress_pct = (total_done / len(all_images)) * 100
        eta_seconds = (total_to_run - processed_this_run) / fps if fps > 0 else 0

        print(
            f"Progress: [{total_done}/{len(all_images)}] ({progress_pct:5.1f}%) "
            f"| Speed: {fps:5.1f} img/s | ETA: {eta_seconds/60:4.1f} min"
        )

    csv_writer.close()
    error_writer.close()
    ambiguous_writer.close()

    total_elapsed = time.time() - start_time
    print(f"\n[DONE] Finished processing in {total_elapsed:.1f}s.")

    print("[STATISTICS] Generating final distributions and summary reports...")
    stats = generate_dataset_statistics(
        str(main_csv_path),
        str(output_dir),
        total_discovered=total_discovered,
        failed_count=failed_this_run,
        skipped_count=skipped_count,
        elapsed_time=total_elapsed,
        device=classifier.clip.device,
        batch_size=args.batch_size
    )

    print("\n" + open(output_dir / "processing_summary.txt", "r", encoding="utf-8").read())


if __name__ == "__main__":
    run()
