import tempfile
import unittest
from pathlib import Path

import numpy as np

from services import ProjectService


class ProjectServiceTests(unittest.TestCase):
    def test_save_project_serializes_numpy_view_state(self):
        service = ProjectService()
        handle = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        handle.close()
        file_path = handle.name
        try:
            service.save_project(
                file_path,
                project_name="demo-project",
                metadata={"priority": np.int64(3)},
                camera_state={
                    "position": np.array([1.0, 2.0, 3.0]),
                    "focal_point": np.array([0.0, 0.0, 0.0]),
                    "view_up": np.array([0.0, 0.0, 1.0]),
                    "distance": np.float64(5.0),
                    "orbit_center": np.array([0.0, 0.0, 0.0]),
                },
                view_state={
                    "projection_mode": True,
                    "workspace_bounds": np.array([-10.0, 10.0, -20.0, 20.0, -5.0, 0.0]),
                },
                objects=[
                    {
                        "object_id": "dataset_1",
                        "parameters": {"origin": np.array([1.0, 2.0, 3.0])},
                    }
                ],
            )

            payload = service.load_project(file_path)

            self.assertEqual(payload["name"], "demo-project")
            self.assertEqual(payload["project_meta"]["name"], "demo-project")
            self.assertEqual(payload["metadata"]["priority"], 3)
            self.assertEqual(payload["camera_state"]["position"], [1.0, 2.0, 3.0])
            self.assertTrue(payload["view_state"]["projection_mode"])
            self.assertEqual(
                payload["view_state"]["workspace_bounds"],
                [-10.0, 10.0, -20.0, 20.0, -5.0, 0.0],
            )
            self.assertEqual(payload["objects"][0]["parameters"]["origin"], [1.0, 2.0, 3.0])
        finally:
            Path(file_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
