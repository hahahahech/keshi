import unittest

import numpy as np

from services import ImportService, SceneService


class DrillholeMappingTests(unittest.TestCase):
    def setUp(self):
        self.import_service = ImportService()
        self.scene_service = SceneService()

    def test_build_well_trajectory_points_from_well_dataset(self):
        well_dataset = self.import_service.load_well_log_dataset(
            "sample_data/sample_borehole_data.csv",
            {
                "x_column": "X",
                "y_column": "Y",
                "z_column": "Z",
                "depth_column": "Depth_m",
                "curve_columns": ["RES_ohm_m"],
                "name": "borehole",
            },
        )
        well_object = self.scene_service.add_dataset(well_dataset, render=False, object_type="dataset")

        points = self.scene_service.build_well_trajectory_points(well_object_id=well_object.object_id)

        self.assertEqual(points.ndim, 2)
        self.assertEqual(points.shape[1], 3)
        self.assertGreaterEqual(points.shape[0], 2)

    def test_create_drillhole_mapping_masks_volume_and_adds_well_objects(self):
        volume_dataset = self.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
        volume_object = self.scene_service.add_dataset(volume_dataset, render=False, object_type="dataset")
        bounds = np.asarray(volume_object.bounds, dtype=float)
        trajectory_points = [
            ((bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, bounds[5]),
            ((bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, bounds[4]),
        ]

        result_objects = self.scene_service.create_drillhole_mapping(
            volume_object.object_id,
            trajectory_points=trajectory_points,
            radius=float((bounds[1] - bounds[0]) / 30.0),
            tube_sides=12,
            render=False,
            add_to_scene=False,
        )

        self.assertGreaterEqual(len(result_objects), 3)
        drilled_volume = next(
            scene_object
            for scene_object in result_objects
            if scene_object.object_type == "clip" and scene_object.source_object_id == volume_object.object_id
        )
        self.assertTrue(drilled_volume.dataset.is_regular_grid)

        scalar_name = drilled_volume.active_scalar
        self.assertIsNotNone(scalar_name)
        values = np.asarray(drilled_volume.data.point_data[scalar_name], dtype=float)
        self.assertTrue(np.isnan(values).any())
        self.assertTrue(np.isfinite(values).any())

        helper_names = {scene_object.name for scene_object in result_objects if scene_object.object_type == "helper"}
        self.assertIn("井轨迹", helper_names)
        self.assertIn("井筒", helper_names)

    def test_create_drillhole_mapping_clips_overlay_surface(self):
        volume_dataset = self.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
        volume_object = self.scene_service.add_dataset(volume_dataset, render=False, object_type="dataset")
        active_scalar = volume_object.active_scalar
        value_range = volume_object.dataset.get_scalar_range(active_scalar)
        isosurface = self.scene_service.create_isosurface(
            volume_object.object_id,
            float((value_range[0] + value_range[1]) / 2.0),
            render=False,
            add_to_scene=True,
        )
        bounds = np.asarray(volume_object.bounds, dtype=float)
        trajectory_points = [
            ((bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, bounds[5]),
            ((bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, bounds[4]),
        ]

        result_objects = self.scene_service.create_drillhole_mapping(
            volume_object.object_id,
            trajectory_points=trajectory_points,
            overlay_object_ids=[isosurface.object_id],
            radius=float((bounds[1] - bounds[0]) / 35.0),
            tube_sides=12,
            render=False,
            add_to_scene=False,
        )

        overlay_results = [
            scene_object
            for scene_object in result_objects
            if scene_object.object_type == "clip" and scene_object.source_object_id == isosurface.object_id
        ]
        self.assertEqual(len(overlay_results), 1)
        self.assertGreater(overlay_results[0].data.n_points, 0)

    def test_create_drillhole_mapping_tube_is_colored_by_source_scalar(self):
        volume_dataset = self.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
        volume_object = self.scene_service.add_dataset(volume_dataset, render=False, object_type="dataset")
        bounds = np.asarray(volume_object.bounds, dtype=float)
        trajectory_points = [
            ((bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, bounds[5]),
            ((bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, bounds[4]),
        ]

        result_objects = self.scene_service.create_drillhole_mapping(
            volume_object.object_id,
            trajectory_points=trajectory_points,
            radius=float((bounds[1] - bounds[0]) / 30.0),
            tube_sides=12,
            render=False,
            add_to_scene=False,
        )

        tube_object = next(
            scene_object
            for scene_object in result_objects
            if scene_object.object_type == "helper"
            and scene_object.parameters.get("kind") == "well_tube"
        )
        self.assertEqual(tube_object.active_scalar, volume_object.active_scalar)
        scalar_name = tube_object.active_scalar
        self.assertIsNotNone(scalar_name)
        has_scalar = scalar_name in tube_object.data.point_data or scalar_name in tube_object.data.cell_data
        self.assertTrue(has_scalar)


if __name__ == "__main__":
    unittest.main()
