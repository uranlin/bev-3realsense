import sys

from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.processing.planner import world_to_grid


def test_offline_demo_to_path():
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/health").json()["depth_model_loaded"] is False
        assert "torch" not in sys.modules
        assert "transformers" not in sys.modules
        assert "pyrealsense2" not in sys.modules
        assert client.get("/").status_code == 200
        scene = client.post("/demo")
        assert scene.status_code == 200
        assert scene.json()["depth_source"] == "synthetic_bev"
        response = client.post("/plan", json={"goal_x": 0, "goal_z": 4})
        assert response.status_code == 200
        result = response.json()
        assert result["success"] and result["path_length_m"] > 3.6
        for point in result["path_world"]:
            rc = world_to_grid(*point)
            assert (
                app.state.runtime.latest_free[rc]
                and not app.state.runtime.latest_inflated[rc]
            )
        assert (
            client.post("/plan", json={"goal_x": 999, "goal_z": 4}).status_code == 422
        )
        assert (
            client.post(
                "/plan", json={"goal_x": 0, "goal_z": 4, "robot_radius": -1}
            ).status_code
            == 422
        )
        assert client.post("/depth_model", json={"model": "outdoor"}).status_code == 409
        assert client.post("/config", json={"depth_max": 0}).status_code == 422
        assert client.post("/config", json={}).status_code == 200
        assert client.post("/plan", json={"goal_x": 0, "goal_z": 4}).status_code == 400


def test_app_factory_owns_separate_runtime():
    first, second = create_app(), create_app()
    with TestClient(first) as a, TestClient(second) as b:
        assert first.state.runtime is not second.state.runtime
        assert a.post("/demo").status_code == 200
        assert b.post("/plan", json={"goal_x": 0, "goal_z": 4}).status_code == 400
        assert a.get("/styles.css").status_code == 200
        assert a.get("/app.js").status_code == 200
