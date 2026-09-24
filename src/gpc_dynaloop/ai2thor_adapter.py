import time
from typing import Any

from gpc_dynaloop.protocol import (
    ActionCommand,
    ActionResult,
    ActionType,
    ObjectObservation,
    Observation,
    TrajectoryStep,
    digest_state,
)


class AI2ThorAdapter:
    def __init__(self, controller):
        self.controller = controller

    def execute(self, command: ActionCommand, step_index: int):
        command.validate()
        started = time.monotonic()
        metadata: dict[str, Any] = {}

        if command.action_type == ActionType.OBSERVE:
            event = self.controller.step(action="Pass")
        elif command.action_type == ActionType.NAVIGATE:
            event, metadata = self._navigate(command.target_object_id)
        elif command.action_type == ActionType.PICKUP:
            event = self.controller.step(
                action="PickupObject",
                objectId=command.target_object_id,
                forceAction=command.parameters.get("force_action", False),
                manualInteract=False,
            )
        elif command.action_type == ActionType.PUT:
            event = self.controller.step(
                action="PutObject",
                objectId=command.target_object_id,
                forceAction=command.parameters.get("force_action", False),
                placeStationary=True,
            )
        elif command.action_type == ActionType.MOVE:
            direction = command.parameters.get("direction", "ahead")
            action = {
                "ahead": "MoveAhead",
                "back": "MoveBack",
                "left": "MoveLeft",
                "right": "MoveRight",
            }[direction]
            event = self.controller.step(
                action=action,
                moveMagnitude=command.parameters.get("magnitude", 0.25),
            )
        elif command.action_type == ActionType.ROTATE:
            direction = command.parameters.get("direction", "right")
            action = "RotateRight" if direction == "right" else "RotateLeft"
            event = self.controller.step(
                action=action,
                degrees=command.parameters.get("degrees", 90),
            )
        elif command.action_type == ActionType.LOOK:
            direction = command.parameters.get("direction", "down")
            action = "LookDown" if direction == "down" else "LookUp"
            event = self.controller.step(
                action=action,
                degrees=command.parameters.get("degrees", 30),
            )
        elif command.action_type == ActionType.END:
            event = self.controller.step(action="Pass")
        else:
            raise ValueError(f"unsupported action: {command.action_type}")

        observation = self.observation(step_index, event)
        result = ActionResult(
            success=event.metadata["lastActionSuccess"],
            error_message=event.metadata.get("errorMessage") or "",
            duration_seconds=round(time.monotonic() - started, 6),
            observation=observation,
            metadata=metadata,
        )
        return TrajectoryStep(step_index=step_index, action=command, result=result)

    def observation(self, step_index: int, event=None):
        event = event or self.controller.last_event
        objects = event.metadata["objects"]
        visible_objects = [
            ObjectObservation(
                object_id=item["objectId"],
                object_type=item["objectType"],
                position=self._vector(item["position"]),
                visible=item["visible"],
                pickupable=item["pickupable"],
                receptacle=item["receptacle"],
                is_picked_up=item["isPickedUp"],
                is_open=item["isOpen"] if item["openable"] else None,
            )
            for item in sorted(objects, key=lambda value: value["objectId"])
            if item["visible"]
        ]
        held_object_ids = sorted(
            item["objectId"] for item in objects if item["isPickedUp"]
        )
        agent = event.metadata["agent"]
        state_digest = digest_state(self.canonical_state(event))
        depth_shape = list(event.depth_frame.shape) if event.depth_frame is not None else None
        observation = Observation(
            step_index=step_index,
            scene_id=event.metadata["sceneName"],
            agent_position=self._vector(agent["position"]),
            agent_rotation=self._vector(agent["rotation"]),
            camera_horizon=round(float(agent["cameraHorizon"]), 6),
            standing=bool(agent["isStanding"]),
            held_object_ids=held_object_ids,
            visible_objects=visible_objects,
            rgb_shape=list(event.frame.shape),
            depth_shape=depth_shape,
            state_digest=state_digest,
        )
        observation.validate()
        return observation

    def canonical_state(self, event=None):
        event = event or self.controller.last_event
        agent = event.metadata["agent"]
        objects = []
        for item in sorted(event.metadata["objects"], key=lambda value: value["objectId"]):
            objects.append(
                {
                    "object_id": item["objectId"],
                    "position": self._vector(item["position"], precision=3),
                    "rotation": self._vector(item["rotation"], precision=3),
                    "is_open": item["isOpen"] if item["openable"] else None,
                    "is_picked_up": item["isPickedUp"],
                    "is_toggled": item["isToggled"] if item["toggleable"] else None,
                }
            )
        return {
            "scene_id": event.metadata["sceneName"],
            "agent": {
                "position": self._vector(agent["position"], precision=3),
                "rotation": self._vector(agent["rotation"], precision=3),
                "camera_horizon": round(float(agent["cameraHorizon"]), 3),
                "standing": bool(agent["isStanding"]),
            },
            "objects": objects,
        }

    def _navigate(self, object_id: str):
        poses_event = self.controller.step(
            action="GetInteractablePoses",
            objectId=object_id,
            standings=[True],
            horizons=[0],
        )
        poses = poses_event.metadata.get("actionReturn") or []
        if not poses:
            return poses_event, {"candidate_pose_count": 0}
        pose = sorted(
            poses,
            key=lambda value: (
                round(value["x"], 6),
                round(value["z"], 6),
                round(value["rotation"], 6),
            ),
        )[0]
        event = self.controller.step(
            action="TeleportFull",
            x=pose["x"],
            y=pose["y"],
            z=pose["z"],
            rotation={"x": 0, "y": pose["rotation"], "z": 0},
            horizon=pose.get("horizon", 0),
            standing=pose.get("standing", True),
        )
        return event, {"candidate_pose_count": len(poses), "selected_pose": pose}

    @staticmethod
    def _vector(value, precision=6):
        return {
            axis: round(float(value[axis]), precision)
            for axis in ("x", "y", "z")
        }
