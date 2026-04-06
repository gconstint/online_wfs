import unittest


class ModuleImportTest(unittest.TestCase):
    def test_calculate_contrast_module_is_importable_from_func_package(self):
        from online_wfs.func import calculate_contrast

        self.assertTrue(callable(calculate_contrast))

    def test_talbot_distance_module_is_importable_from_func_package(self):
        from online_wfs.func import calculate_spherical_wave_talbot_distance

        self.assertTrue(callable(calculate_spherical_wave_talbot_distance))

    def test_package_exports_func_subpackage(self):
        import online_wfs

        self.assertTrue(hasattr(online_wfs, "func"))


if __name__ == "__main__":
    unittest.main()
