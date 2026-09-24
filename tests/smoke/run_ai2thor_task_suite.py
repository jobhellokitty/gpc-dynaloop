import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering


SCENES = ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4"]
OUTPUT_PATH = Path(
    "/root/pc/gpc_dynaloop_storage/results/stage1/ai2thor_smoke_20.json"
)


def result(scene, task, passed, detail, elapsed):
    return {
        "scene": scene,
        "task": task,
        "passed": passed,
        "detail": detail,
        "elapsed_seconds": round(elapsed, 3),
    }


def main():
    started = time.monotonic()
    results = []
    controller = None

    try:
        controller = Controller(
            scene=SCENES[0],
            platform=CloudRendering,
            width=300,
            height=300,
            quality="Low",
            renderDepthImage=True,
        )

        for scene in SCENES:
            scene_started = time.monotonic()
            controller.reset(scene)
            event = controller.last_event
            initialized = (
                event.metadata["lastActionSuccess"]
                and event.frame.shape == (300, 300, 3)
                and event.depth_frame.shape == (300, 300)
            )
            results.append(
                result(
                    scene,
                    "initialize_and_observe",
                    initialized,
                    f"objects={len(event.metadata['objects'])}",
                    time.monotonic() - scene_started,
                )
            )

            task_started = time.monotonic()
            reachable_event = controller.step(action="GetReachablePositions")
            positions = reachable_event.metadata.get("actionReturn") or []
            reachable_ok = reachable_event.metadata["lastActionSuccess"] and bool(positions)
            results.append(
                result(
                    scene,
                    "reachable_positions",
                    reachable_ok,
                    f"count={len(positions)}",
                    time.monotonic() - task_started,
                )
            )

            task_started = time.monotonic()
            rotations = [controller.step(action="RotateRight") for _ in range(4)]
            rotation_ok = all(item.metadata["lastActionSuccess"] for item in rotations)
            results.append(
                result(
                    scene,
                    "rotate_roundtrip",
                    rotation_ok,
                    "steps=4",
                    time.monotonic() - task_started,
                )
            )

            task_started = time.monotonic()
            look_down = controller.step(action="LookDown")
            look_up = controller.step(action="LookUp")
            look_ok = (
                look_down.metadata["lastActionSuccess"]
                and look_up.metadata["lastActionSuccess"]
            )
            results.append(
                result(
                    scene,
                    "look_roundtrip",
                    look_ok,
                    "steps=2",
                    time.monotonic() - task_started,
                )
            )

            task_started = time.monotonic()
            if positions:
                position = positions[len(positions) // 2]
                teleport = controller.step(
                    action="TeleportFull",
                    x=position["x"],
                    y=position["y"],
                    z=position["z"],
                    rotation={"x": 0, "y": 0, "z": 0},
                    horizon=0,
                    standing=True,
                )
                teleport_ok = teleport.metadata["lastActionSuccess"]
                teleport_detail = teleport.metadata.get("errorMessage") or "target=reachable"
            else:
                teleport_ok = False
                teleport_detail = "no reachable target"
            results.append(
                result(
                    scene,
                    "teleport_to_reachable",
                    teleport_ok,
                    teleport_detail,
                    time.monotonic() - task_started,
                )
            )
    finally:
        if controller is not None:
            controller.stop()

    passed = sum(item["passed"] for item in results)
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "success_rate": passed / len(results) if results else 0,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "results": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print(f"total: {report['total']}")
    print(f"passed: {report['passed']}")
    print(f"failed: {report['failed']}")
    print(f"success_rate: {report['success_rate']:.2%}")
    print(f"elapsed_seconds: {report['elapsed_seconds']}")
    print(f"report: {OUTPUT_PATH}")

    if report["total"] != 20 or report["failed"]:
        raise SystemExit(1)
    print("AI2THOR_20_TASK_SMOKE_OK")


if __name__ == "__main__":
    main()
