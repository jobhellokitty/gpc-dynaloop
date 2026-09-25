import json
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info


def build_step_entries(record, manifest_entry):
    entries = []
    image_count = 0
    for index, message in enumerate(record["messages"]):
        if message["role"] == "user":
            image_count += message.get("content", "").count("<image>")
        if message["role"] != "assistant":
            continue
        if index == 0 or record["messages"][index - 1]["role"] != "user":
            raise ValueError(f"assistant message {index} does not follow a user message")
        entries.append(
            {
                "id": f"{manifest_entry['split']}:{manifest_entry['source_index']}:{index}",
                "split": manifest_entry["split"],
                "source_index": manifest_entry["source_index"],
                "target_message_index": index,
                "image_count": image_count,
                "task_type": manifest_entry["task_type"],
                "scene": manifest_entry["scene"],
            }
        )
    return entries


def _multimodal_content(text, image_paths, image_cursor):
    parts = text.split("<image>")
    content = []
    for index, part in enumerate(parts):
        if index:
            if image_cursor >= len(image_paths):
                raise ValueError("message contains more image tokens than image paths")
            content.append({"type": "image", "image": str(image_paths[image_cursor])})
            image_cursor += 1
        if part:
            content.append({"type": "text", "text": part})
    return content, image_cursor


def materialize_step(record, entry, dataset_root):
    target_index = entry["target_message_index"]
    messages = record["messages"][: target_index + 1]
    image_paths = [
        Path(dataset_root) / image.removeprefix("./") for image in record["images"]
    ]
    converted = []
    image_cursor = 0
    for message in messages:
        content = message["content"]
        if message["role"] == "user":
            content, image_cursor = _multimodal_content(
                content, image_paths, image_cursor
            )
        converted.append({"role": message["role"], "content": content})
    if image_cursor != entry["image_count"]:
        raise ValueError(
            f"materialized {image_cursor} images, expected {entry['image_count']}"
        )
    return {"id": entry["id"], "messages": converted}


def build_sft_inputs(record, processor):
    messages = record["messages"]
    prompt_messages = messages[:-1]
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs = process_vision_info(messages)
    full = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    prompt = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    labels = full["input_ids"].clone()
    prompt_length = int(prompt["attention_mask"].sum().item())
    if not torch.equal(
        prompt["input_ids"][0, :prompt_length],
        full["input_ids"][0, :prompt_length],
    ):
        raise ValueError("prompt tokens are not an exact prefix of the conversation")
    labels[:, :prompt_length] = -100
    labels[full["attention_mask"] == 0] = -100
    if not torch.any(labels != -100):
        raise ValueError("assistant target has no trainable tokens")
    full["labels"] = labels
    full["prompt_length"] = torch.tensor(prompt_length)
    return full


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
