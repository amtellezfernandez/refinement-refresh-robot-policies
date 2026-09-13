"""Offline tests for confirmation artifact safety and numerical presentation."""

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from analyze_confirmation import extract_snapshot, latex_table, number


class ConfirmationArtifactTests(unittest.TestCase):
    def test_percentage_rounding(self):
        self.assertEqual(number(76.25), "76.3")
        self.assertEqual(number(6.249999999999999, signed=True), "+6.3")
        self.assertEqual(number(-6.25, signed=True), "-6.3")
        self.assertEqual(number(6.874999999999999, signed=True), "+6.9")

    def test_latex_rows(self):
        content = latex_table("rr", "A & B", [["1", "2"], ["3", "4"]])
        self.assertIn("A & B" + r"\\", content)
        self.assertIn("1 & 2" + r"\\", content)
        self.assertTrue(content.endswith("\\end{tabular}\n"))

    def extraction(self, name, member_type=tarfile.REGTYPE, duplicate=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot.tar.gz"
            with tarfile.open(snapshot, "w:gz") as archive:
                for _ in range(2 if duplicate else 1):
                    member = tarfile.TarInfo(name)
                    member.type = member_type
                    if member.isfile():
                        member.size = 2
                        archive.addfile(member, io.BytesIO(b"{}"))
                    else:
                        member.linkname = "../outside"
                        archive.addfile(member)
            destination = root / "extracted"
            destination.mkdir()
            extract_snapshot(snapshot, destination)
            self.assertEqual((destination / name).read_text(), "{}")

    def test_regular_member(self):
        self.extraction("phase/result.json")

    def test_reject_traversal(self):
        with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
            self.extraction("../outside.json")

    def test_reject_absolute(self):
        with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
            self.extraction("/outside.json")

    def test_reject_links(self):
        for kind in [tarfile.SYMTYPE, tarfile.LNKTYPE]:
            with (
                self.subTest(kind=kind),
                self.assertRaisesRegex(ValueError, "Unsafe archive member"),
            ):
                self.extraction("link", kind)

    def test_reject_duplicate(self):
        with self.assertRaisesRegex(ValueError, "Duplicate archive member"):
            self.extraction("phase/result.json", duplicate=True)


if __name__ == "__main__":
    unittest.main()
