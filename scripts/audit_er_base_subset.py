import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

from gpc_dynaloop.er_subset import select_records, write_jsonl


CONFIG_PATH = Path("/root/pc/gpc_dynaloop/configs/er_base_subset.json")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_inventory(repo):
    response = requests.get(
        f"https://huggingface.co/api/datasets/{repo}",
        params={"blobs": "true"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    files = {
        item["rfilename"]: {
            "size": item.get("size"),
            "sha256": (item.get("lfs") or {}).get("sha256"),
        }
        for item in payload["siblings"]
    }
    return payload["sha"], files


def probe_range_support(repo, revision, path):
    response = requests.head(
        f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}",
        allow_redirects=True,
        timeout=60,
    )
    response.raise_for_status()
    return {
        "path": path,
        "status_code": response.status_code,
        "accept_ranges": response.headers.get("accept-ranges"),
        "content_length": int(response.headers["content-length"]),
        "etag": response.headers.get("etag"),
    }


def main():
    config = json.loads(CONFIG_PATH.read_text())
    metadata_root = Path(config["local_metadata_root"])
    output_root = Path(config["output_root"])
    manifest_root = output_root / "manifests"
    train_path = metadata_root / "train_multiturn_9390.json"
    records = json.loads(train_path.read_text())

    selected, issues, task_stats = select_records(records, config)
    for split, items in selected.items():
        write_jsonl(manifest_root / f"{split}.jsonl", items)
    for stale_name in ("val.jsonl", "official_test.jsonl"):
        stale_path = manifest_root / stale_name
        if stale_path.exists():
            stale_path.unlink()

    revision, remote_files = remote_inventory(config["source_repo"])
    selected_items = [item for items in selected.values() for item in items]
    selected_archives = sorted({item["archive_path"] for item in selected_items})
    missing_archives = [path for path in selected_archives if path not in remote_files]
    selected_image_counts = Counter()
    for item in selected_items:
        selected_image_counts[item["task_type"]] += item["image_count"]

    archive_plan = []
    archive_total = 0
    selective_estimate = 0
    for archive_path in selected_archives:
        task_type = Path(archive_path).stem
        archive_size = remote_files[archive_path]["size"]
        source_images = task_stats[task_type]["images"]
        selected_images = selected_image_counts[task_type]
        estimated_bytes = round(archive_size * selected_images / source_images)
        archive_total += archive_size
        selective_estimate += estimated_bytes
        archive_plan.append(
            {
                "task_type": task_type,
                "archive_path": archive_path,
                "archive_bytes": archive_size,
                "archive_sha256": remote_files[archive_path]["sha256"],
                "source_image_count": source_images,
                "selected_image_count": selected_images,
                "selective_download_estimate_bytes": estimated_bytes,
            }
        )

    range_probe = probe_range_support(
        config["source_repo"], revision, selected_archives[0]
    )

    disk = shutil.disk_usage(output_root.parent)
    split_summary = {
        split: {
            "record_count": len(items),
            "image_count": sum(item["image_count"] for item in items),
            "scene_count": len({item["scene"] for item in items}),
            "task_counts": dict(sorted(Counter(item["task_type"] for item in items).items())),
        }
        for split, items in selected.items()
    }
    scene_sets = {split: {item["scene"] for item in items} for split, items in selected.items()}
    split_overlap = {
        "train_test": sorted(scene_sets["train"] & scene_sets["test"]),
    }
    upstream_commit = subprocess.check_output(
        ["git", "-C", "/root/pc/Embodied-Omni", "rev-parse", "HEAD"], text=True
    ).strip()
    checks = {
        "source_record_count_9390": len(records) == 9390,
        "metadata_records_valid": not issues,
        "all_configured_task_types_present": all(
            any(
                item["task_type"] == task_type
                for items in selected.values()
                for item in items
            )
            for task_type in config["task_types"]
        ),
        "train_and_test_nonempty": all(selected.values()),
        "scene_splits_disjoint": not any(split_overlap.values()),
        "all_archives_exist_remotely": not missing_archives,
        "selective_range_supported": range_probe["accept_ranges"] == "bytes"
        and range_probe["content_length"] == remote_files[range_probe["path"]]["size"],
        "storage_supports_archive_strategy": disk.free >= archive_total + 20 * 1024**3,
        "no_training_images_downloaded": not any(output_root.rglob("*.png"))
        and not any(output_root.rglob("*.zip")),
    }
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "subset_name": config["subset_name"],
        "passed": all(checks.values()),
        "checks": checks,
        "provenance": {
            "embodied_omni_commit": upstream_commit,
            "huggingface_repo": config["source_repo"],
            "huggingface_revision": revision,
            "train_metadata_sha256": sha256(train_path),
        },
        "split_summary": split_summary,
        "issues": issues[:100],
        "issue_count": len(issues),
        "split_overlap": split_overlap,
        "missing_archives": missing_archives,
        "download_budget": {
            "strategy_recommended": "selective_zip_range",
            "selected_archive_count": len(selected_archives),
            "full_archive_download_bytes": archive_total,
            "selective_download_estimate_bytes": selective_estimate,
            "free_disk_bytes": disk.free,
            "largest_archive_bytes": max(item["archive_bytes"] for item in archive_plan),
            "range_probe": range_probe,
        },
        "archive_plan": archive_plan,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "remote_files.json").write_text(
        json.dumps({"revision": revision, "files": remote_files}, indent=2) + "\n"
    )
    (output_root / "download_plan.json").write_text(
        json.dumps(
            {
                "revision": revision,
                "strategy": "selective_zip_range",
                "archives": archive_plan,
            },
            indent=2,
        )
        + "\n"
    )
    (output_root / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
    print("ER_BASE_SUBSET_PREDOWNLOAD_OK")


if __name__ == "__main__":
    main()
