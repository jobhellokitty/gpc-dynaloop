import json
from datetime import datetime, timezone
from pathlib import Path


RESULTS_DIR = Path("/root/pc/gpc_dynaloop_storage/results/stage1")


def main():
    protocol = json.loads((RESULTS_DIR / "stage1_protocol_suite.json").read_text())
    model = json.loads((RESULTS_DIR / "gpc_multi_image/report.json").read_text())
    checks = {
        "protocol_tasks_20_of_20": protocol["task_total"] == 20
        and protocol["task_passed"] == 20,
        "determinism_100_percent": protocol["determinism_rate"] == 1.0,
        "gpc_sft_he_loaded": model["model_class"]
        == "Qwen2_5_VLForConditionalGeneration",
        "multi_image_count_two": model["image_count"] == 2,
        "generation_nonempty": model["output_nonempty"],
    }
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "protocol_report": str(RESULTS_DIR / "stage1_protocol_suite.json"),
        "model_report": str(RESULTS_DIR / "gpc_multi_image/report.json"),
    }
    output = RESULTS_DIR / "stage1_gate.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
    print("STAGE1_GATE_OK")


if __name__ == "__main__":
    main()
