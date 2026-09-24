from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering


def main():
    controller = None
    try:
        controller = Controller(
            scene="FloorPlan1",
            platform=CloudRendering,
            width=300,
            height=300,
            quality="Low",
            renderDepthImage=True,
        )

        initial = controller.last_event
        assert initial.metadata["lastActionSuccess"], initial.metadata.get("errorMessage")
        assert initial.frame.shape == (300, 300, 3)
        assert initial.depth_frame.shape == (300, 300)

        rotated = controller.step(action="RotateRight")
        assert rotated.metadata["lastActionSuccess"], rotated.metadata.get("errorMessage")

        reachable = controller.step(action="GetReachablePositions")
        assert reachable.metadata["lastActionSuccess"], reachable.metadata.get("errorMessage")

        print("scene:", initial.metadata["sceneName"])
        print("rgb_frame:", initial.frame.shape)
        print("depth_frame:", initial.depth_frame.shape)
        print("objects:", len(initial.metadata["objects"]))
        print("reachable_positions:", len(reachable.metadata["actionReturn"]))
        print("AI2THOR_CLOUDRENDERING_SMOKE_OK")
    finally:
        if controller is not None:
            controller.stop()


if __name__ == "__main__":
    main()
