import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from gpc_dynaloop.ai2thor_adapter import AI2ThorAdapter
from gpc_dynaloop.protocol import (
    ActionCommand,
    ActionType,
    SceneSpec,
    Trajectory,
    digest_state,
)


SCENES = ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4"]
SEED = 20260924
OUTPUT_DIR = Path("/root/pc/gpc_dynaloop_storage/results/stage1")


def command(scene, index, action_type, target=None, parameters=None):
    return ActionCommand(
        action_id=f"{scene}-{index}-{action_type.value}",
        action_type=action_type,
        target_object_id=target,
        parameters=parameters or {},
    )


def select_pick_and_put_targets(controller):
    objects = sorted(controller.last_event.metadata["objects"], key=lambda item: item["objectId"])
    object_map = {item["objectId"]: item for item in objects}
    for item in objects:
        parents = item.get("parentReceptacles") or []
        if not item["pickupable"] or not parents:
            continue
        receptacles = [parent for parent in parents if object_map.get(parent, {}).get("receptacle")]
        if not receptacles:
            continue
        pickup_poses = controller.step(
            action="GetInteractablePoses",
            objectId=item["objectId"],
            standings=[True],
            horizons=[0],
        ).metadata.get("actionReturn") or []
        if not pickup_poses:
            continue
        for receptacle_id in receptacles:
            put_poses = controller.step(
                action="GetInteractablePoses",
                objectId=receptacle_id,
                standings=[True],
                horizons=[0],
            ).metadata.get("actionReturn") or []
            if put_poses:
                return item["objectId"], receptacle_id
    raise RuntimeError("no deterministic pickup/put target pair found")


def rotation_delta(left, right):
    difference = abs(left - right) % 360
    return min(difference, 360 - difference)


def compare_states(left, right):
    position_tolerance = 0.01
    rotation_tolerance = 0.1
    mismatches = []
    max_position_delta = 0.0
    max_rotation_delta = 0.0

    if left["scene_id"] != right["scene_id"]:
        mismatches.append("scene_id")
    if left["agent"]["standing"] != right["agent"]["standing"]:
        mismatches.append("agent.standing")
    if left["agent"]["camera_horizon"] != right["agent"]["camera_horizon"]:
        mismatches.append("agent.camera_horizon")

    left_objects = {item["object_id"]: item for item in left["objects"]}
    right_objects = {item["object_id"]: item for item in right["objects"]}
    if left_objects.keys() != right_objects.keys():
        mismatches.append("object_ids")

    entities = [("agent", left["agent"], right["agent"])]
    entities.extend(
        (object_id, left_objects[object_id], right_objects[object_id])
        for object_id in sorted(left_objects.keys() & right_objects.keys())
    )
    for entity_id, left_item, right_item in entities:
        for axis in ("x", "y", "z"):
            position_delta = abs(left_item["position"][axis] - right_item["position"][axis])
            rotation_difference = rotation_delta(
                left_item["rotation"][axis], right_item["rotation"][axis]
            )
            max_position_delta = max(max_position_delta, position_delta)
            max_rotation_delta = max(max_rotation_delta, rotation_difference)
        for field in ("is_open", "is_picked_up", "is_toggled"):
            if field in left_item and left_item[field] != right_item[field]:
                mismatches.append(f"{entity_id}.{field}")

    return {
        "matched": (
            not mismatches
            and max_position_delta <= position_tolerance
            and max_rotation_delta <= rotation_tolerance
        ),
        "position_tolerance": position_tolerance,
        "rotation_tolerance": rotation_tolerance,
        "max_position_delta": round(max_position_delta, 6),
        "max_rotation_delta": round(max_rotation_delta, 6),
        "semantic_mismatches": mismatches,
    }


def deterministic_check(controller, adapter, scene):
    states = []
    for _ in range(2):
        controller.reset(scene)
        event = controller.step(
            action="InitialRandomSpawn",
            randomSeed=SEED,
            forceVisible=False,
            numPlacementAttempts=5,
            placeStationary=True,
        )
        if not event.metadata["lastActionSuccess"]:
            raise RuntimeError(event.metadata.get("errorMessage"))
        for _ in range(3):
            event = controller.step(action="Pass")
        states.append(adapter.canonical_state(event))
    comparison = compare_states(states[0], states[1])
    comparison["digests"] = [digest_state(state) for state in states]
    return comparison


def main():
    started = time.monotonic()
    controller = None
    trajectories = []
    determinism = []

    try:
        controller = Controller(
            scene=SCENES[0],
            platform=CloudRendering,
            width=300,
            height=300,
            quality="Low",
            renderDepthImage=True,
        )
        adapter = AI2ThorAdapter(controller)

        for scene in SCENES:
            comparison = deterministic_check(controller, adapter, scene)
            determinism.append({"scene": scene, "seed": SEED, **comparison})

            controller.reset(scene)
            pickup_id, receptacle_id = select_pick_and_put_targets(controller)
            trajectory = Trajectory(
                trajectory_id=f"stage1-{scene}",
                scene=SceneSpec(scene_id=scene, seed=SEED),
                task_instruction=f"Pick up {pickup_id} and put it in or on {receptacle_id}.",
                metadata={
                    "reproduction_level": "method adaptation",
                    "pickup_object_id": pickup_id,
                    "receptacle_object_id": receptacle_id,
                },
            )
            actions = [
                command(scene, 0, ActionType.OBSERVE),
                command(scene, 1, ActionType.NAVIGATE, pickup_id),
                command(
                    scene,
                    2,
                    ActionType.PICKUP,
                    pickup_id,
                    {"force_action": True},
                ),
                command(scene, 3, ActionType.NAVIGATE, receptacle_id),
                command(
                    scene,
                    4,
                    ActionType.PUT,
                    receptacle_id,
                    {"force_action": True},
                ),
            ]
            for index, action in enumerate(actions):
                trajectory.append(adapter.execute(action, index))
            trajectory.validate()
            trajectories.append(trajectory)
    finally:
        if controller is not None:
            controller.stop()

    steps = [step for trajectory in trajectories for step in trajectory.steps]
    passed = sum(step.result.success for step in steps)
    deterministic_passed = sum(item["matched"] for item in determinism)
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0.0",
        "reproduction_level": "method adaptation",
        "task_total": len(steps),
        "task_passed": passed,
        "task_failed": len(steps) - passed,
        "task_success_rate": passed / len(steps),
        "determinism_total": len(determinism),
        "determinism_passed": deterministic_passed,
        "determinism_rate": deterministic_passed / len(determinism),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "determinism": determinism,
        "trajectories": [trajectory.to_dict() for trajectory in trajectories],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "stage1_protocol_suite.json"
    trajectory_path = OUTPUT_DIR / "stage1_trajectories.jsonl"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    trajectory_path.write_text(
        "".join(trajectory.to_json() + "\n" for trajectory in trajectories)
    )

    print(f"task_total: {report['task_total']}")
    print(f"task_passed: {report['task_passed']}")
    print(f"task_failed: {report['task_failed']}")
    print(f"task_success_rate: {report['task_success_rate']:.2%}")
    print(f"determinism_rate: {report['determinism_rate']:.2%}")
    for item in determinism:
        if not item["matched"]:
            print(
                "determinism_failure:",
                item["scene"],
                f"max_position_delta={item['max_position_delta']}",
                f"max_rotation_delta={item['max_rotation_delta']}",
                f"semantic_mismatches={item['semantic_mismatches']}",
            )
    print(f"elapsed_seconds: {report['elapsed_seconds']}")
    print(f"report: {report_path}")
    print(f"trajectories: {trajectory_path}")

    if len(steps) != 20 or passed != 20 or deterministic_passed != len(determinism):
        raise SystemExit(1)
    print("STAGE1_PROTOCOL_SUITE_OK")


if __name__ == "__main__":
    main()
