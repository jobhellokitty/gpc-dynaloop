import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


IMAGE_PATTERN = re.compile(r"(?:\./)?data/images/([^/]+)/([^/]+)/(.+)")
SCENE_PATTERN = re.compile(r"FloorPlan(\d+)")


def parse_image_path(path):
    match = IMAGE_PATTERN.fullmatch(path)
    if not match:
        raise ValueError(f"unsupported Embodied-Reasoner image path: {path}")
    task_type, trajectory_name, filename = match.groups()
    scene_match = SCENE_PATTERN.search(trajectory_name)
    if not scene_match:
        raise ValueError(f"scene missing from trajectory path: {path}")
    return {
        "task_type": task_type,
        "trajectory_name": trajectory_name,
        "filename": filename,
        "scene": f"FloorPlan{scene_match.group(1)}",
    }


def scene_split(scene, split_config):
    match = SCENE_PATTERN.fullmatch(scene)
    if not match:
        raise ValueError(f"unsupported scene: {scene}")
    offset = (int(match.group(1)) - 1) % 100 + 1
    train_start, train_end = split_config["train_offsets"]
    test_start, test_end = split_config["test_offsets"]
    if train_start <= offset <= train_end:
        return "train"
    if test_start <= offset <= test_end:
        return "test"
    raise ValueError(f"scene offset outside configured split: {scene}")


def image_token_count(record):
    return sum(
        message.get("content", "").count("<image>")
        for message in record.get("messages", [])
        if message.get("role") == "user"
    )


def record_digest(record):
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def build_candidates(records, config):
    candidates = defaultdict(list)
    issues = []
    task_stats = defaultdict(lambda: {"records": 0, "images": 0, "scenes": set()})
    configured_types = set(config["task_types"])

    for source_index, record in enumerate(records):
        images = record.get("images") or []
        if not images:
            issues.append({"source_index": source_index, "issue": "missing_images"})
            continue
        try:
            parsed = [parse_image_path(path) for path in images]
        except ValueError as error:
            issues.append({"source_index": source_index, "issue": str(error)})
            continue
        task_type = parsed[0]["task_type"]
        scene = parsed[0]["scene"]
        if any(item["task_type"] != task_type or item["scene"] != scene for item in parsed):
            issues.append({"source_index": source_index, "issue": "mixed_task_or_scene"})
            continue
        if image_token_count(record) != len(images):
            issues.append({"source_index": source_index, "issue": "image_token_mismatch"})
            continue

        stats = task_stats[task_type]
        stats["records"] += 1
        stats["images"] += len(images)
        stats["scenes"].add(scene)
        if task_type not in configured_types:
            continue
        split = scene_split(scene, config["scene_split"])
        item = {
            "source_index": source_index,
            "source_file": "train_multiturn_9390.json",
            "task_type": task_type,
            "scene": scene,
            "split": split,
            "trajectory_name": parsed[0]["trajectory_name"],
            "archive_path": f"data/images/{task_type}.zip",
            "images": images,
            "image_count": len(images),
            "record_sha256": record_digest(record),
        }
        candidates[(task_type, split)].append(item)

    for stats in task_stats.values():
        stats["scenes"] = sorted(stats["scenes"])
        stats["scene_count"] = len(stats["scenes"])
    return candidates, issues, dict(task_stats)


def select_records(records, config):
    candidates, issues, task_stats = build_candidates(records, config)
    selected = {"train": [], "test": []}
    for task_type in config["task_types"]:
        for split in selected:
            selected[split].extend(candidates[(task_type, split)])
    for split in selected:
        selected[split].sort(key=lambda item: (item["task_type"], item["scene"], item["source_index"]))
    return selected, issues, task_stats


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
