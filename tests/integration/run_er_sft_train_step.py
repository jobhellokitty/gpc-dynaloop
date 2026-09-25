import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from gpc_dynaloop.er_sft import build_sft_inputs, materialize_step, read_jsonl


DATASET_ROOT = Path(
    "/root/pc/gpc_dynaloop_storage/datasets/er_base_subset_v1"
)
SOURCE_PATH = Path(
    "/root/pc/Embodied-Omni/embodied_reasoner/data/train_multiturn_9390.json"
)
MODEL_PATH = Path("/root/pc/three_model_release_v1/work/models/GPC-SFT-HE")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260925)
    torch.cuda.manual_seed_all(20260925)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    source_records = json.loads(SOURCE_PATH.read_text())
    entries = read_jsonl(DATASET_ROOT / "sft_index" / "train.jsonl")
    entry = min(entries, key=lambda item: (item["image_count"], item["source_index"]))
    record = materialize_step(
        source_records[entry["source_index"]], entry, DATASET_ROOT
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True, use_fast=False
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()
    base.config.use_cache = False
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    base.enable_input_require_grads()
    model = get_peft_model(
        base,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-4,
    )
    inputs = build_sft_inputs(record, processor)
    inputs.pop("prompt_length")
    inputs = {
        key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    loss = model(**inputs).loss
    if not torch.isfinite(loss):
        raise RuntimeError("training loss is not finite")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        max_norm=1.0,
    )
    if not torch.isfinite(grad_norm):
        raise RuntimeError("gradient norm is not finite")
    optimizer.step()
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    report = {
        "sample_id": entry["id"],
        "loss": float(loss.detach().item()),
        "gradient_norm_before_clipping": float(grad_norm.item()),
        "trainable_parameters": trainable,
        "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("ER_SFT_TRAIN_STEP_OK")


if __name__ == "__main__":
    main()
