import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_PATH = Path("/root/pc/three_model_release_v1/work/models/GPC-SFT-HE")
OUTPUT_DIR = Path("/root/pc/gpc_dynaloop_storage/results/stage1/gpc_multi_image")


def capture_images():
    controller = None
    try:
        controller = Controller(
            scene="FloorPlan1",
            platform=CloudRendering,
            width=300,
            height=300,
            quality="Low",
            renderDepthImage=False,
        )
        first = controller.last_event
        second = controller.step(action="RotateRight")
        if not first.metadata["lastActionSuccess"] or not second.metadata["lastActionSuccess"]:
            raise RuntimeError("failed to capture deterministic AI2-THOR observations")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        first_path = OUTPUT_DIR / "observation_0.png"
        second_path = OUTPUT_DIR / "observation_1.png"
        Image.fromarray(first.frame).save(first_path)
        Image.fromarray(second.frame).save(second_path)
        return first_path, second_path
    finally:
        if controller is not None:
            controller.stop()


def main():
    started = time.monotonic()
    image_paths = capture_images()
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Compare consecutive embodied observations and return a concise structured answer.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_paths[0])},
                {"type": "image", "image": str(image_paths[1])},
                {
                    "type": "text",
                    "text": (
                        "The second observation was captured after one camera action. "
                        "What changed between the two views? Return exactly "
                        "<process>[{\"step\":1,\"operation\":\"compare\",\"result\":\"...\"}]"
                        "</process><answer>...</answer>."
                    ),
                },
            ],
        },
    ]

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    image_count = int(inputs["image_grid_thw"].shape[0])
    if image_count != 2:
        raise RuntimeError(f"expected two images, processor produced {image_count}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda().eval()
    cuda_inputs = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    prompt_length = cuda_inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **cuda_inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
        )
    output = processor.batch_decode(
        generated[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    success = bool(output)
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(MODEL_PATH),
        "model_class": type(model).__name__,
        "image_count": image_count,
        "input_token_count": int(prompt_length),
        "output": output,
        "output_nonempty": success,
        "structured_tags_present": "<process>" in output and "<answer>" in output,
        "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    report_path = OUTPUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not success:
        raise SystemExit(1)
    print("GPC_SFT_HE_MULTI_IMAGE_OK")


if __name__ == "__main__":
    main()
