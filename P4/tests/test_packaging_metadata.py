import pathlib
import tomllib
import unittest


class PackagingMetadataTests(unittest.TestCase):
    def test_project_license_is_not_declared_as_a_console_script(self):
        pyproject_path = pathlib.Path(__file__).parents[1] / "pyproject.toml"
        with pyproject_path.open("rb") as stream:
            metadata = tomllib.load(stream)

        project = metadata["project"]
        self.assertEqual(project["version"], "0.3.0")
        self.assertEqual(project["dependencies"], [])
        self.assertEqual(project["optional-dependencies"]["protocol"], ["torch>=2.1"])
        self.assertEqual(project["license"], {"text": "MIT"})
        license_path = pyproject_path.parent / "LICENSE"
        self.assertIn("MIT License", license_path.read_text(encoding="utf-8"))
        self.assertTrue(
            all(isinstance(entrypoint, str) for entrypoint in project["scripts"].values())
        )


if __name__ == "__main__":
    unittest.main()
