import unittest

from gpc_dynaloop.protocol import (
    ActionCommand,
    ActionResult,
    ActionType,
    Observation,
    SceneSpec,
    Trajectory,
    TrajectoryStep,
    digest_state,
)


def observation(step_index=0):
    return Observation(
        step_index=step_index,
        scene_id="FloorPlan1_physics",
        agent_position={"x": 0.0, "y": 0.9, "z": 0.0},
        agent_rotation={"x": 0.0, "y": 0.0, "z": 0.0},
        camera_horizon=0.0,
        standing=True,
        held_object_ids=[],
        visible_objects=[],
        rgb_shape=[300, 300, 3],
        depth_shape=[300, 300],
        state_digest=digest_state({"state": 1}),
    )


class ProtocolTest(unittest.TestCase):
    def test_valid_trajectory_serializes(self):
        trajectory = Trajectory(
            trajectory_id="trajectory-1",
            scene=SceneSpec(scene_id="FloorPlan1", seed=7),
            task_instruction="Observe the room.",
        )
        action = ActionCommand(action_id="action-0", action_type=ActionType.OBSERVE)
        trajectory.append(
            TrajectoryStep(
                step_index=0,
                action=action,
                result=ActionResult(
                    success=True,
                    error_message="",
                    duration_seconds=0.1,
                    observation=observation(),
                ),
            )
        )
        payload = trajectory.to_dict()
        self.assertEqual(payload["schema_version"], "1.0.0")
        self.assertEqual(len(payload["steps"]), 1)

    def test_targeted_action_requires_target(self):
        command = ActionCommand(action_id="action-0", action_type=ActionType.PICKUP)
        with self.assertRaises(ValueError):
            command.validate()

    def test_steps_must_be_contiguous(self):
        trajectory = Trajectory(
            trajectory_id="trajectory-1",
            scene=SceneSpec(scene_id="FloorPlan1", seed=7),
            task_instruction="Observe the room.",
        )
        action = ActionCommand(action_id="action-1", action_type=ActionType.OBSERVE)
        step = TrajectoryStep(
            step_index=1,
            action=action,
            result=ActionResult(
                success=True,
                error_message="",
                duration_seconds=0.1,
                observation=observation(1),
            ),
        )
        with self.assertRaises(ValueError):
            trajectory.append(step)

    def test_digest_is_key_order_independent(self):
        self.assertEqual(digest_state({"a": 1, "b": 2}), digest_state({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
