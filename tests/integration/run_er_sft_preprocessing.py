import json
from pathlib import Path

from transformers import AutoProcessor

from gpc_dynaloop.er_sft import build_sft_inputs, materialize_step, read_jsonl


DATASET_ROOT = Path(
    "/root/pc/gpc_dynaloop_storage/datasets/er_base_subset_v1"
)
SOURCE_PATH = Path(
    "/root/pc/Embodied-Omni/embodied_reasoner/data/train_multiturn_9390.json"
)
MODEL_PATH = Path("/root/pc/three_model_release_v1/work/models/GPC-SFT-HE")


def main():
    source_records = json.loads(SOURCE_PATH.read_text())
    entries = read_jsonl(DATASET_ROOT / "sft_index" / "train.jsonl")
    entry = min(entries, key=lambda item: (item["image_count"], item["source_index"]))
    record = materialize_step(
        source_records[entry["source_index"]], entry, DATASET_ROOT
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True, use_fast=False
    )
    inputs = build_sft_inputs(record, processor)
    trainable_tokens = int((inputs["labels"] != -100).sum().item())
    report = {
        "sample_id": entry["id"],
        "task_type": entry["task_type"],
        "image_count": entry["image_count"],
        "input_tokens": int(inputs["attention_mask"].sum().item()),
        "trainable_tokens": trainable_tokens,
        "pixel_values_shape": list(inputs["pixel_values"].shape),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if trainable_tokens <= 0 or entry["image_count"] <= 0:
        raise SystemExit(1)
    print("ER_SFT_PREPROCESSING_OK")


if __name__ == "__main__":
    main()
