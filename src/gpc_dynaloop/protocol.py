import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


SCHEMA_VERSION = "1.0.0"


class ActionType(str, Enum):
    OBSERVE = "observe"
    NAVIGATE = "navigate"
    PICKUP = "pickup"
    PUT = "put"
    MOVE = "move"
    ROTATE = "rotate"
    LOOK = "look"
    END = "end"


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    seed: int
    simulator: str = "AI2-THOR"
    simulator_version: str = "5.0.0"

    def validate(self):
        if not self.scene_id:
            raise ValueError("scene_id is required")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class ActionCommand:
    action_id: str
    action_type: ActionType
    target_object_id: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def validate(self):
        if not self.action_id:
            raise ValueError("action_id is required")
        if self.action_type in {
            ActionType.NAVIGATE,
            ActionType.PICKUP,
            ActionType.PUT,
        } and not self.target_object_id:
            raise ValueError(f"{self.action_type.value} requires target_object_id")


@dataclass(frozen=True)
class ObjectObservation:
    object_id: str
    object_type: str
    position: dict[str, float]
    visible: bool
    pickupable: bool
    receptacle: bool
    is_picked_up: bool
    is_open: bool | None


@dataclass(frozen=True)
class Observation:
    step_index: int
    scene_id: str
    agent_position: dict[str, float]
    agent_rotation: dict[str, float]
    camera_horizon: float
    standing: bool
    held_object_ids: list[str]
    visible_objects: list[ObjectObservation]
    rgb_shape: list[int]
    depth_shape: list[int] | None
    state_digest: str

    def validate(self):
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if not self.scene_id:
            raise ValueError("observation scene_id is required")
        if len(self.rgb_shape) != 3:
            raise ValueError("rgb_shape must contain three dimensions")
        if len(self.state_digest) != 64:
            raise ValueError("state_digest must be a SHA-256 hex digest")


@dataclass(frozen=True)
class ActionResult:
    success: bool
    error_message: str
    duration_seconds: float
    observation: Observation
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryStep:
    step_index: int
    action: ActionCommand
    result: ActionResult

    def validate(self):
        self.action.validate()
        self.result.observation.validate()
        if self.step_index != self.result.observation.step_index:
            raise ValueError("step and observation indices must match")


@dataclass
class Trajectory:
    trajectory_id: str
    scene: SceneSpec
    task_instruction: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, step: TrajectoryStep):
        if step.step_index != len(self.steps):
            raise ValueError("trajectory step indices must be contiguous")
        step.validate()
        self.steps.append(step)

    def validate(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if not self.trajectory_id:
            raise ValueError("trajectory_id is required")
        if not self.task_instruction:
            raise ValueError("task_instruction is required")
        self.scene.validate()
        action_ids = set()
        for index, step in enumerate(self.steps):
            if step.step_index != index:
                raise ValueError("trajectory step indices must be contiguous")
            if step.action.action_id in action_ids:
                raise ValueError("action_id must be unique within a trajectory")
            action_ids.add(step.action.action_id)
            step.validate()

    def to_dict(self):
        self.validate()
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def digest_state(state: dict[str, Any]):
    payload = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
