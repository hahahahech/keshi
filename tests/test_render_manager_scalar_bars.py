import unittest

import pyvista as pv

from rendering.render_manager import RenderManager
from services import ImportService, SceneService


class RenderManagerScalarBarTests(unittest.TestCase):
    def setUp(self):
        self.plotter = pv.Plotter(off_screen=True)
        self.render_manager = RenderManager(self.plotter)
        self.scene_service = SceneService(render_manager=self.render_manager)
        self.import_service = ImportService()

    def tearDown(self):
        self.plotter.close()

    def test_only_one_scalar_bar_is_kept_across_objects(self):
        first = self.scene_service.add_dataset(
            self.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv"),
            render=True,
            name="first",
        )
        second = self.scene_service.add_dataset(
            self.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv"),
            render=True,
            name="second",
        )

        self.scene_service.update_style(second.object_id, scalar_name="density_inverted")

        self.assertEqual(list(self.plotter.scalar_bars.keys()), ["density_inverted"])
        self.assertEqual(self.render_manager._scalar_bar_owner_id, second.object_id)
        self.assertIsNotNone(first)

    def test_turning_off_scalar_bar_clears_owned_bar(self):
        scene_object = self.scene_service.add_dataset(
            self.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv"),
            render=True,
        )
        self.assertEqual(list(self.plotter.scalar_bars.keys()), ["density_true"])

        self.scene_service.update_style(scene_object.object_id, show_scalar_bar=False)

        self.assertEqual(list(self.plotter.scalar_bars.keys()), [])
        self.assertIsNone(self.render_manager._scalar_bar_owner_id)


if __name__ == "__main__":
    unittest.main()
