import unittest

from services import ImportService, SceneService


class SceneServiceSliceTests(unittest.TestCase):
    def setUp(self):
        self.import_service = ImportService()
        self.scene_service = SceneService()

    def test_orthogonal_slice_supports_regular_grid_multiblock_result(self):
        dataset = self.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
        scene_object = self.scene_service.add_dataset(dataset, render=False)
        bounds = scene_object.bounds

        derived = self.scene_service.create_orthogonal_slice(
            scene_object.object_id,
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
            render=False,
            add_to_scene=False,
        )

        self.assertEqual(derived.object_type, "slice")
        self.assertGreater(derived.data.n_points, 0)
        self.assertGreater(derived.data.n_cells, 0)

    def test_axis_slice_batch_skips_empty_boundary_positions(self):
        dataset = self.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
        scene_object = self.scene_service.add_dataset(dataset, render=False)
        bounds = scene_object.bounds

        derived = self.scene_service.create_axis_slice_batch(
            scene_object.object_id,
            "x",
            bounds[0],
            bounds[1],
            (bounds[1] - bounds[0]) / 10.0,
            render=False,
            add_to_scene=False,
        )

        self.assertEqual(len(derived), 10)
        for item in derived:
            self.assertEqual(item.object_type, "slice")
            self.assertGreater(item.data.n_points, 0)
            self.assertGreater(item.data.n_cells, 0)

    def test_polyline_section_creates_continuous_vertical_fence(self):
        dataset = self.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
        scene_object = self.scene_service.add_dataset(dataset, render=False)

        derived = self.scene_service.create_polyline_section(
            scene_object.object_id,
            [
                (100.0, 120.0, 0.0),
                (520.0, 260.0, 0.0),
                (880.0, 860.0, 0.0),
            ],
            top_z=0.0,
            bottom_z=-500.0,
            line_step=60.0,
            vertical_samples=16,
            render=False,
            add_to_scene=False,
        )

        self.assertEqual(derived.object_type, "section")
        self.assertGreater(derived.data.n_points, 0)
        self.assertGreater(derived.data.n_cells, 0)
        self.assertIn("density_true", list(derived.data.point_data.keys()))


if __name__ == "__main__":
    unittest.main()
