import hashlib
import json
import subprocess
from pathlib import Path

from gpc_dynaloop.er_sft import read_jsonl


DATASET_ROOT = Path(
    "/root/pc/gpc_dynaloop_storage/datasets/er_base_subset_v1"
)
SOURCE_PATH = Path(
    "/root/pc/Embodied-Omni/embodied_reasoner/data/train_multiturn_9390.json"
)
LLAMAFACTORY_ROOT = Path("/root/pc/LLaMA-Factory-Embodied")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    source_records = json.loads(SOURCE_PATH.read_text())
    output_root = DATASET_ROOT / "llamafactory"
    output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "source_path": str(SOURCE_PATH),
        "source_sha256": sha256(SOURCE_PATH),
        "embodied_reasoner_commit": subprocess.check_output(
            ["git", "-C", "/root/pc/Embodied-Omni", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "llamafactory_commit": subprocess.check_output(
            ["git", "-C", str(LLAMAFACTORY_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "splits": {},
    }
    dataset_info = {}

    for split in ("train", "test"):
        manifest = read_jsonl(DATASET_ROOT / "manifests" / f"{split}.jsonl")
        records = [source_records[item["source_index"]] for item in manifest]
        output_path = output_root / f"{split}.json"
        output_path.write_text(json.dumps(records, ensure_ascii=False) + "\n")
        name = f"er_base_{split}"
        dataset_info[name] = {
            "file_name": output_path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
        report["splits"][split] = {
            "trajectory_count": len(records),
            "file": str(output_path),
            "sha256": sha256(output_path),
        }

    (output_root / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n"
    )
    (output_root / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("ER_LLAMAFACTORY_EXPORT_OK")


if __name__ == "__main__":
    main()
