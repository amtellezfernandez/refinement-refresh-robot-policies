"""Read-only runtime selection checks; no model/framework imports or identities."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import verify_campaign as V


class RuntimeRoles(unittest.TestCase):
    def test_server_rejects_client_python_before_any_artifact_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'absent'
            with patch.object(V.sys, 'version_info', (3, 8, 20)):
                with self.assertRaisesRegex(ValueError, 'Python >=3.11'):
                    V.verify(root, 'pi05', role='server')
            self.assertFalse(root.exists())

    def test_minimum_version_is_role_specific(self):
        V.check_runtime_role('client', (3, 8, 20))
        V.check_runtime_role('server', (3, 11, 0))
        V.check_runtime_role('server', (3, 12, 0))
        with self.assertRaises(ValueError):
            V.check_runtime_role('server', (3, 10, 15))

    def test_native_helper_remains_python38_syntax(self):
        ast.parse(Path(V.__file__).read_text(), feature_version=(3, 8))


if __name__ == '__main__':
    unittest.main()
