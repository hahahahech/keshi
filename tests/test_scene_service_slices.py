import unittest

import numpy as np

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

        self.assertEqual(derived.object_type, "slice")
        self.assertEqual(derived.parameters.get("kind"), "polyline")
        self.assertGreater(derived.data.n_points, 0)
        self.assertGreater(derived.data.n_cells, 0)
        self.assertIn("density_true", list(derived.data.point_data.keys()))

    def test_move_polyline_slice_offsets_polyline_points(self):
        dataset = self.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
        source = self.scene_service.add_dataset(dataset, render=False)
        section = self.scene_service.create_polyline_section(
            source.object_id,
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
            add_to_scene=True,
        )

        moved = self.scene_service.move_slice(
            section.object_id,
            80.0,
            render=False,
            add_to_scene=False,
        )

        self.assertEqual(moved.object_type, "slice")
        self.assertEqual(moved.parameters.get("kind"), "polyline")
        original_points = np.asarray(section.parameters["points"], dtype=float)
        moved_points = np.asarray(moved.parameters["points"], dtype=float)
        self.assertEqual(original_points.shape, moved_points.shape)
        self.assertGreater(float(np.linalg.norm(moved_points - original_points)), 1e-6)

    def test_tilt_polyline_slice_rotates_points_in_xy(self):
        dataset = self.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
        source = self.scene_service.add_dataset(dataset, render=False)
        section = self.scene_service.create_polyline_section(
            source.object_id,
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
            add_to_scene=True,
        )

        tilted = self.scene_service.tilt_slice(
            section.object_id,
            angle_deg=20.0,
            tilt_axis="z",
            render=False,
            add_to_scene=False,
        )

        self.assertEqual(tilted.object_type, "slice")
        self.assertEqual(tilted.parameters.get("kind"), "polyline")
        original_points = np.asarray(section.parameters["points"], dtype=float)
        tilted_points = np.asarray(tilted.parameters["points"], dtype=float)
        self.assertEqual(original_points.shape, tilted_points.shape)
        self.assertGreater(float(np.linalg.norm(tilted_points[:, :2] - original_points[:, :2])), 1e-6)

    def test_polyline_plane_slice_creates_slice_object(self):
        dataset = self.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
        scene_object = self.scene_service.add_dataset(dataset, render=False)

        derived = self.scene_service.create_polyline_plane_slice(
            scene_object.object_id,
            [
                (100.0, 120.0, 0.0),
                (520.0, 260.0, 0.0),
                (880.0, 860.0, 0.0),
            ],
            render=False,
            add_to_scene=False,
        )

        self.assertEqual(derived.object_type, "slice")
        self.assertEqual(derived.parameters.get("kind"), "plane")
        self.assertEqual(derived.parameters.get("line_mode"), "polyline_plane")
        self.assertGreater(derived.data.n_points, 0)
        self.assertGreater(derived.data.n_cells, 0)

    def test_clip_box_inherits_source_render_mode(self):
        dataset = self.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
        scene_object = self.scene_service.add_dataset(dataset, render=False)
        self.scene_service.update_style(scene_object.object_id, render_mode="volume")
        bounds = scene_object.bounds

        derived = self.scene_service.create_clip_box(
            scene_object.object_id,
            (
                bounds[0],
                (bounds[0] + bounds[1]) / 2.0,
                bounds[2],
                bounds[3],
                bounds[4],
                bounds[5],
            ),
            render=False,
            add_to_scene=False,
        )

        self.assertEqual(derived.object_type, "clip")
        self.assertEqual(derived.render_mode, "volume")
        self.assertTrue(derived.dataset.is_regular_grid)

    def test_grid_index_clip_creates_regular_grid_clip(self):
        dataset = self.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
        scene_object = self.scene_service.add_dataset(dataset, render=False)
        dims = tuple(int(v) for v in scene_object.data.dimensions)

        derived = self.scene_service.create_grid_index_clip(
            scene_object.object_id,
            (0, max(1, dims[0] // 2), 0, max(1, dims[1] // 2), 0, dims[2] - 1),
            render=False,
            add_to_scene=False,
        )

        self.assertEqual(derived.object_type, "clip")
        self.assertTrue(derived.dataset.is_regular_grid)
        self.assertEqual(derived.render_mode, scene_object.render_mode)
        self.assertGreater(derived.data.n_points, 0)

    def test_grid_index_clip_rejects_point_set_dataset(self):
        dataset = self.import_service.load_dataset("sample_data/synthetic_forward_points.xyz")
        scene_object = self.scene_service.add_dataset(dataset, render=False)

        with self.assertRaises(ValueError):
            self.scene_service.create_grid_index_clip(
                scene_object.object_id,
                (0, 1, 0, 1, 0, 1),
                render=False,
                add_to_scene=False,
            )

    def test_mask_clip_from_polyline_keeps_regular_grid_and_render_mode(self):
        dataset = self.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
        scene_object = self.scene_service.add_dataset(dataset, render=False)
        self.scene_service.update_style(scene_object.object_id, render_mode="volume")
        bounds = scene_object.bounds
        mid_x = (bounds[0] + bounds[1]) / 2.0
        mid_y = (bounds[2] + bounds[3]) / 2.0
        points = [
            (bounds[0], bounds[2], bounds[5]),
            (mid_x, bounds[2], bounds[5]),
            (mid_x, mid_y, bounds[5]),
            (bounds[0], mid_y, bounds[5]),
        ]

        derived = self.scene_service.create_mask_clip_from_polyline(
            scene_object.object_id,
            points,
            render=False,
            add_to_scene=False,
        )

        self.assertEqual(derived.object_type, "clip")
        self.assertTrue(derived.dataset.is_regular_grid)
        self.assertEqual(derived.render_mode, "volume")
        scalar_name = derived.active_scalar
        values = np.asarray(derived.data.point_data[scalar_name], dtype=float)
        self.assertTrue(np.isnan(values).any())
        self.assertTrue(np.isfinite(values).any())

    def test_move_slice_translates_along_slice_normal(self):
        dataset = self.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
        source = self.scene_service.add_dataset(dataset, render=False)
        bounds = source.bounds
        position = (bounds[4] + bounds[5]) / 2.0
        slice_object = self.scene_service.create_axis_slice(
            source.object_id,
            "z",
            position,
            render=False,
            add_to_scene=True,
        )

        moved = self.scene_service.move_slice(
            slice_object.object_id,
            120.0,
            render=False,
            add_to_scene=False,
        )

        self.assertEqual(moved.object_type, "slice")
        self.assertEqual(moved.parameters.get("kind"), "plane")
        moved_origin = np.asarray(moved.parameters["origin"], dtype=float)
        moved_normal = np.asarray(moved.parameters["normal"], dtype=float)
        np.testing.assert_allclose(moved_normal, np.array([0.0, 0.0, 1.0]), atol=1e-6)
        self.assertAlmostEqual(float(moved_origin[2]), float(position + 120.0), places=5)
        self.assertGreater(moved.data.n_points, 0)

    def test_tilt_slice_rotates_normal_and_keeps_slice_non_empty(self):
        dataset = self.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
        source = self.scene_service.add_dataset(dataset, render=False)
        bounds = source.bounds
        position = (bounds[4] + bounds[5]) / 2.0
        slice_object = self.scene_service.create_axis_slice(
            source.object_id,
            "z",
            position,
            render=False,
            add_to_scene=True,
        )

        tilted = self.scene_service.tilt_slice(
            slice_object.object_id,
            angle_deg=30.0,
            tilt_axis="x",
            render=False,
            add_to_scene=False,
        )

        self.assertEqual(tilted.object_type, "slice")
        self.assertEqual(tilted.parameters.get("kind"), "plane")
        normal = np.asarray(tilted.parameters["normal"], dtype=float)
        self.assertGreater(np.linalg.norm(normal), 0.99)
        self.assertLess(float(normal[2]), 0.999)
        self.assertGreater(abs(float(normal[1])), 1e-6)
        self.assertGreater(tilted.data.n_points, 0)


if __name__ == "__main__":
    unittest.main()
