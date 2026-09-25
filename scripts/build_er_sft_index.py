import json
from collections import Counter
from pathlib import Path

from gpc_dynaloop.er_sft import build_step_entries, read_jsonl


DATASET_ROOT = Path(
    "/root/pc/gpc_dynaloop_storage/datasets/er_base_subset_v1"
)
SOURCE_PATH = Path(
    "/root/pc/Embodied-Omni/embodied_reasoner/data/train_multiturn_9390.json"
)


def main():
    source_records = json.loads(SOURCE_PATH.read_text())
    output_root = DATASET_ROOT / "sft_index"
    output_root.mkdir(parents=True, exist_ok=True)
    report = {"source_path": str(SOURCE_PATH), "splits": {}}

    for split in ("train", "test"):
        manifest = read_jsonl(DATASET_ROOT / "manifests" / f"{split}.jsonl")
        entries = []
        for manifest_entry in manifest:
            record = source_records[manifest_entry["source_index"]]
            entries.extend(build_step_entries(record, manifest_entry))
        output_path = output_root / f"{split}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        report["splits"][split] = {
            "trajectory_count": len(manifest),
            "step_count": len(entries),
            "task_counts": dict(sorted(Counter(e["task_type"] for e in entries).items())),
            "max_context_images": max(e["image_count"] for e in entries),
        }

    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("ER_SFT_INDEX_OK")


if __name__ == "__main__":
    main()
