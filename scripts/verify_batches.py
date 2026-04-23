"""
Task 8: Verify Test Set Batch Organization
- Confirms test set is correctly split into N_BATCHES batches
- Checks each batch contains all 4 class subdirectories
- Validates image counts per batch per class (should equal BATCH_SIZE)
- Detects any augmented images that leaked into the test set
- Detects cross-batch duplicate filenames (same image in 2+ batches)
- Cross-checks disk state against the split manifest if available
- Saves a verification report to outputs/metrics/batch_verification.json

Run from project root after split_dataset.py:
  python scripts/verify_batches.py
"""

import os
import sys
import json

TEST_DIR      = "data/processed/test"
MANIFEST_PATH = "outputs/metrics/split_manifest.json"
REPORT_PATH   = "outputs/metrics/batch_verification.json"
CLASSES       = ["glioma", "meningioma", "no_tumor", "pituitary"]
N_BATCHES     = 4
BATCH_SIZE    = 100   # images per class per batch


# --------------------------------------------------------------------------
# Core helpers
# --------------------------------------------------------------------------

def _count_images(directory: str) -> tuple[int, int]:
    """
    Return (readable_count, aug_count) for PNG files in a flat directory.
    aug_count flags augmented images that should NOT be in the test set.
    """
    if not os.path.exists(directory):
        return 0, 0
    readable = 0
    aug      = 0
    for fname in os.listdir(directory):
        if fname.lower().endswith(".png"):
            if "_aug" in fname:
                aug += 1
            else:
                readable += 1
    return readable, aug


def verify_class_counts(
    test_dir:   str = TEST_DIR,
    n_batches:  int = N_BATCHES,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """
    Verify image counts per class per batch.

    Args:
        test_dir:   Root test directory containing batch subdirs.
        n_batches:  Expected number of batch directories.
        batch_size: Expected images per class per batch.

    Returns:
        Dict with per-batch, per-class counts and any discrepancies found.
        Keys: 'batches', 'errors', 'warnings', 'pass'
    """
    result = {
        "test_dir":   test_dir,
        "n_batches":  n_batches,
        "batch_size": batch_size,
        "batches":    {},
        "errors":     [],
        "warnings":   [],
        "pass":       True,
    }

    if not os.path.exists(test_dir):
        result["errors"].append(f"Test directory not found: {test_dir}")
        result["pass"] = False
        return result

    for b in range(1, n_batches + 1):
        batch_name = f"batch{b}"
        batch_dir  = os.path.join(test_dir, batch_name)
        batch_info = {"classes": {}, "status": "ok"}

        if not os.path.exists(batch_dir):
            result["errors"].append(f"Missing batch directory: {batch_dir}")
            result["pass"] = False
            batch_info["status"] = "missing"
            result["batches"][batch_name] = batch_info
            continue

        for cls in CLASSES:
            cls_dir = os.path.join(batch_dir, cls)
            readable, aug = _count_images(cls_dir)

            cls_info = {
                "path":     cls_dir,
                "count":    readable,
                "aug_leak": aug,
            }

            # Wrong count
            if readable != batch_size:
                msg = (f"{batch_name}/{cls}: expected {batch_size} images, "
                       f"got {readable}")
                result["errors"].append(msg)
                result["pass"] = False
                cls_info["issue"] = "wrong_count"

            # Augmented images in test (data leakage)
            if aug > 0:
                msg = (f"{batch_name}/{cls}: {aug} augmented image(s) found "
                       f"— these should not be in the test set")
                result["errors"].append(msg)
                result["pass"] = False
                cls_info["issue"] = cls_info.get("issue", "") + " aug_in_test"

            # Missing class directory
            if not os.path.exists(cls_dir):
                result["errors"].append(f"Missing class dir: {cls_dir}")
                result["pass"] = False
                cls_info["issue"] = "missing_dir"

            batch_info["classes"][cls] = cls_info

        result["batches"][batch_name] = batch_info

    return result


def _detect_cross_batch_duplicates(test_dir: str, n_batches: int) -> list[str]:
    """
    Return list of (class, filename) pairs that appear in more than one batch.
    A same-class, same-filename collision across batches indicates a split error.

    Note: The same filename appearing in DIFFERENT classes is legitimate
    (e.g., 'img_001.png' can exist in both glioma/ and meningioma/).
    Only same-class duplicates across batches are flagged.
    """
    # key: (cls, fname) → list of batch names where it appears
    seen: dict[tuple[str, str], list[str]] = {}
    for b in range(1, n_batches + 1):
        batch_name = f"batch{b}"
        for cls in CLASSES:
            cls_dir = os.path.join(test_dir, batch_name, cls)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(".png"):
                    key = (cls, fname)
                    seen.setdefault(key, []).append(batch_name)

    return [
        f"{cls}/{fname} in multiple batches: {', '.join(batches)}"
        for (cls, fname), batches in seen.items()
        if len(batches) > 1
    ]


def _cross_check_manifest(test_dir: str, manifest_path: str, n_batches: int) -> list[str]:
    """
    Compare files on disk against the split manifest.
    Returns list of discrepancy messages (empty = fully consistent).
    """
    if not os.path.exists(manifest_path):
        return ["Manifest not found — skipping manifest cross-check"]

    with open(manifest_path) as f:
        manifest = json.load(f)

    discrepancies = []
    for cls, data in manifest.get("classes", {}).items():
        if data.get("status") != "ok":
            continue
        for batch_name, manifest_files in data.get("test_batches", {}).items():
            disk_dir   = os.path.join(test_dir, batch_name, cls)
            disk_files = set()
            if os.path.exists(disk_dir):
                disk_files = {
                    f for f in os.listdir(disk_dir) if f.endswith(".png")
                }
            manifest_set = set(manifest_files)

            only_on_disk     = disk_files - manifest_set
            only_in_manifest = manifest_set - disk_files

            if only_on_disk:
                discrepancies.append(
                    f"{batch_name}/{cls}: {len(only_on_disk)} file(s) on disk "
                    f"not in manifest: {sorted(only_on_disk)[:3]}"
                )
            if only_in_manifest:
                discrepancies.append(
                    f"{batch_name}/{cls}: {len(only_in_manifest)} file(s) in "
                    f"manifest missing from disk: {sorted(only_in_manifest)[:3]}"
                )

    return discrepancies


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def verify_batches(
    test_dir:      str = TEST_DIR,
    manifest_path: str = MANIFEST_PATH,
    n_batches:     int = N_BATCHES,
    batch_size:    int = BATCH_SIZE,
) -> dict:
    """
    Run the full batch verification suite and save a JSON report.

    Returns:
        Full report dict with counts, errors, warnings, and overall pass/fail.
    """
    print("=" * 60)
    print("  TASK 8: TEST BATCH VERIFICATION")
    print("=" * 60)
    print(f"  Test dir  : {test_dir}")
    print(f"  Batches   : {n_batches}  |  Per class per batch: {batch_size}")
    print(f"  Classes   : {', '.join(CLASSES)}")
    print("-" * 60)

    # 1. Count check
    report = verify_class_counts(test_dir, n_batches, batch_size)

    # Print per-batch summary
    total_images = 0
    for batch_name, binfo in report["batches"].items():
        batch_total = sum(
            v["count"] for v in binfo.get("classes", {}).values()
        )
        total_images += batch_total
        status_icon = "✅" if binfo["status"] == "ok" and batch_name not in [
            e.split(":")[0].split("/")[0] for e in report["errors"]
        ] else "❌"
        print(f"  {status_icon}  {batch_name}: {batch_total} images "
              f"({', '.join(str(binfo['classes'].get(c, {}).get('count', '?')) for c in CLASSES)})")

    # 2. Cross-batch duplicate check
    print()
    duplicates = _detect_cross_batch_duplicates(test_dir, n_batches)
    if duplicates:
        print(f"  ❌ Cross-batch duplicates ({len(duplicates)}):")
        for d in duplicates[:5]:
            print(f"     {d}")
        report["errors"].extend(duplicates)
        report["pass"] = False
    else:
        print("  ✅ Cross-batch duplicates : none")

    report["cross_batch_duplicates"] = duplicates

    # 3. Manifest cross-check
    manifest_issues = _cross_check_manifest(test_dir, manifest_path, n_batches)
    if manifest_issues and manifest_issues != ["Manifest not found — skipping manifest cross-check"]:
        print(f"  ❌ Manifest discrepancies ({len(manifest_issues)}):")
        for m in manifest_issues[:5]:
            print(f"     {m}")
        report["errors"].extend(manifest_issues)
        report["pass"] = False
    elif manifest_issues:
        print(f"  ⚠  {manifest_issues[0]}")
        report["warnings"].extend(manifest_issues)
    else:
        print("  ✅ Manifest cross-check   : consistent")

    report["manifest_issues"] = manifest_issues

    # 4. Summary
    print()
    print("-" * 60)
    print(f"  Total images in test set: {total_images}")
    print(f"  Expected               : {n_batches * batch_size * len(CLASSES)}")
    print()
    if report["errors"]:
        print(f"  ❌ FAILED — {len(report['errors'])} error(s) found:")
        for err in report["errors"]:
            print(f"     • {err}")
    else:
        print("  ✅ ALL CHECKS PASSED")
    print("=" * 60)

    # Save report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {REPORT_PATH}")

    return report


if __name__ == "__main__":
    if not os.path.exists(TEST_DIR):
        print(f"ERROR: Test directory '{TEST_DIR}' not found.")
        print("  Ensure split_dataset.py has been run first.")
        sys.exit(1)

    report = verify_batches()
    sys.exit(0 if report["pass"] else 1)
