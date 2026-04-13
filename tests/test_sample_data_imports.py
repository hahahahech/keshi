import unittest

from services import ImportService


class SampleDataImportTests(unittest.TestCase):
    def setUp(self):
        self.import_service = ImportService()

    def test_geologic_contrast_model_imports_as_regular_grid(self):
        for path in (
            "sample_data/geologic_contrast_model.csv",
            "sample_data/geologic_contrast_model.vtr",
        ):
            dataset = self.import_service.load_dataset(path)
            self.assertEqual(dataset.dataset_kind, "regular_grid")
            self.assertEqual(dataset.metadata.dataset_type, "regular_grid")
            self.assertIn("lithology_index", dataset.scalar_names)
            self.assertIn("display_contrast_index", dataset.scalar_names)


if __name__ == "__main__":
    unittest.main()
