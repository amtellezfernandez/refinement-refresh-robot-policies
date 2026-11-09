"""Presentation regressions: no refitting or modification of evidence reports."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import render_paper_visuals as visuals


class PaperVisualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = {
            name: json.loads((visuals.CAMPAIGN / path).read_text())
            for name, path in visuals.SOURCES.items()
        }

    def test_report_validation_is_read_only(self):
        before = copy.deepcopy(self.data)
        visuals.validate(self.data)
        self.assertEqual(before, self.data)

    def test_rounding_removes_binary_noise_at_exact_half_ties(self):
        self.assertEqual(visuals.number(6.249999999999999), "6.3")
        self.assertEqual(visuals.number(16.875), "16.9")
        self.assertEqual(visuals.number(-2.5, True), "-2.5")

    def test_resource_coordinates_have_different_feasible_refreshes(self):
        self.assertEqual(
            visuals.feasible_prefixes(self.data["calibration"]), [[5, 2, 1], [2, 1, 1]]
        )

    def test_primary_counts_and_costs_are_retained(self):
        text = visuals.tables(self.data)["confirmation_resources.tex"]
        self.assertIn("2 & 2 & 139/160 & 86.9 & 0.429 & 41.1 & 149.0", text)
        self.assertIn("2 & 5 & 122/160 & 76.3 & 0.455 & 48.7 & 166.3", text)
        self.assertIn("5 & 2 & 138/160 & 86.3 & 0.171 & 16.1 & 144.9", text)
        self.assertEqual(text.count(r"\addlinespace"), 2)

    def test_all_intervals_and_analysis_groups_are_retained(self):
        tables = visuals.tables(self.data)
        self.assertIn(r"$[+4.4,\,+16.9]$", tables["confirmation_predictions.tex"])
        self.assertIn(r"$[-6.3,\,+6.9]$", tables["confirmation_secondary.tex"])
        # Exactly three frozen and four amended rows, not selected by interval.
        for name, expected in (
            ("confirmation_predictions.tex", 3),
            ("confirmation_secondary.tex", 4),
        ):
            body = tables[name].split(r"\midrule")[1].split(r"\bottomrule")[0]
            self.assertEqual(sum(" & " in line for line in body.splitlines()), expected)

    def test_submitted_primary_resource_intervals_are_used(self):
        text = visuals.tables(self.data)["primary_resources.tex"]
        self.assertIn(r"41.1 & 48.7 & -7.6 & $[-10.6,\,-4.5]$", text)
        self.assertIn(r"19.0 & 34.3 & -15.2 & $[-17.2,\,-13.3]$", text)

    def test_inconsistent_counts_fail(self):
        data = copy.deepcopy(self.data)
        data["confirmation"]["resources"][3]["successes"] -= 1
        with self.assertRaisesRegex(ValueError, "Task counts"):
            visuals.validate(data)

    def test_inconsistent_primary_interval_fails(self):
        data = copy.deepcopy(self.data)
        data["primary"]["metrics"]["success"]["descriptive_95_interval"][0] = 0
        with self.assertRaisesRegex(ValueError, "primary interval"):
            visuals.validate(data)

    def test_all_tables_use_decimal_alignment_and_booktabs(self):
        for content in visuals.tables(self.data).values():
            self.assertIn("S[table-format=", content)
            self.assertIn(r"\toprule", content)
            self.assertIn(r"\bottomrule", content)
            self.assertNotIn(r"\resizebox", content)
            self.assertNotIn(r"\scriptsize", content)

    def test_task_totals_match_the_overview(self):
        text = visuals.tables(self.data)["confirmation_tasks.tex"]
        self.assertIn("Total (/160) & 137 & 136 & 135 & 139 & 122 & 138 & 133", text)
        self.assertIn("S[table-format=3.0]", text)
        self.assertEqual(text.count(r"\midrule"), 2)

    def test_discovery_plot_keeps_every_estimate_and_interval(self):
        import numpy as np

        visuals.style()
        with patch.object(visuals, "save_figure") as save:
            visuals.discovery_figure(self.data, Path("unused.pdf"))
        figure = save.call_args.args[0]
        axis, rule = figure.axes
        self.assertEqual(len(axis.containers), 3)
        for h, container in zip((1, 2, 5), axis.containers):
            rows = [r for r in self.data["discovery"]["cells"] if r["h"] == h]
            np.testing.assert_allclose(
                container.lines[0].get_ydata(), [100 * r["success"] for r in rows]
            )
            segments = container.lines[2][0].get_segments()
            self.assertEqual(len(segments), 4)
            for row, segment in zip(rows, segments):
                np.testing.assert_allclose(
                    segment[:, 1], 100 * np.array(row["descriptive_95_interval"])
                )
        self.assertEqual(axis.get_ylim(), (0, 100))
        self.assertIn("Rule-selected", rule.get_ylabel())
        np.testing.assert_equal(rule.lines[0].get_ydata(), [0, 1, 2])
        np.testing.assert_equal(rule.lines[1].get_ydata(), [1, 2, 2])

    def test_confirmation_plot_keeps_all_seven_paired_intervals(self):
        import numpy as np

        visuals.style()
        with patch.object(visuals, "save_figure") as save:
            visuals.confirmation_figure(self.data, Path("unused.pdf"))
        axis = save.call_args.args[0].axes[0]
        self.assertEqual(len(axis.containers), 7)
        for i, (comparison, container) in enumerate(
            zip(self.data["confirmation"]["comparisons"], axis.containers)
        ):
            np.testing.assert_allclose(
                np.asarray(container.lines[0].get_xdata(), dtype=float),
                [100 * comparison["mean_difference"][0]],
            )
            segment = container.lines[2][0].get_segments()[0]
            np.testing.assert_allclose(
                segment[:, 0], 100 * np.array(comparison["interval95"][0])
            )
            self.assertEqual(
                container.lines[0].get_marker(),
                "D" if i == 1 else ("o" if i < 3 else "s"),
            )

    def test_shared_cost_figure_uses_the_retained_affine_decomposition(self):
        import numpy as np
        from matplotlib.text import Text

        visuals.style()
        with patch.object(visuals, "save_figure") as save:
            visuals.cost_decomposition_figure(self.data, Path("unused.pdf"))
        figure = save.call_args.args[0]
        axis = figure.axes[0]
        base = self.data["calibration"]["affine"]["base"] / 1e9
        np.testing.assert_allclose(
            [bar.get_height() for bar in axis.patches[:4]], [base] * 4
        )
        settings = (2, 5, 10, 20)
        totals = [
            self.data["calibration"]["generation_flops"][str(d)] / 1e9
            for d in settings
        ]
        np.testing.assert_allclose(
            [bar.get_height() for bar in axis.patches[4:]],
            np.asarray(totals) - base,
        )
        text = " ".join(item.get_text() for item in figure.findobj(Text))
        self.assertIn("82.1% of native", text)
        self.assertIn("756.2", text)

    def test_shared_tradeoff_figure_keeps_all_settings_and_paired_intervals(self):
        import numpy as np
        from matplotlib.text import Text

        visuals.style()
        with patch.object(visuals, "save_figure") as save:
            visuals.success_compute_figure(self.data, Path("unused.pdf"))
        figure = save.call_args.args[0]
        tradeoff, effects = figure.axes
        points = np.vstack([collection.get_offsets()[0] for collection in tradeoff.collections])
        expected = np.array(
            [
                [r["nominal_fraction_native"], 100 * r["success_rate"]]
                for r in self.data["confirmation"]["resources"]
            ]
        )
        np.testing.assert_allclose(points, expected)
        self.assertEqual(len(effects.containers), 2)
        np.testing.assert_allclose(
            np.asarray(effects.containers[0].lines[0].get_xdata(), dtype=float),
            [10.625],
        )
        np.testing.assert_allclose(
            np.asarray(effects.containers[1].lines[0].get_xdata(), dtype=float),
            [-0.625],
        )
        text = " ".join(item.get_text() for item in figure.findobj(Text))
        self.assertIn("+10.6 [+4.4, +16.9]", text)
        self.assertIn("-0.6 [-6.9, +6.3]", text)
        self.assertIn("41.1 → 16.1 TFLOPs", text)
        self.assertIn("Equivalence is not established", text)

    def test_task_figure_uses_reported_counts_without_invented_intervals(self):
        import numpy as np
        from matplotlib.text import Text

        visuals.style()
        with patch.object(visuals, "save_figure") as save:
            visuals.task_level_figure(self.data, Path("unused.pdf"))
        figure = save.call_args.args[0]
        refinement = figure.axes[0].collections[0].get_offsets()[:, 0]
        refresh = figure.axes[1].collections[0].get_offsets()[:, 0]
        np.testing.assert_equal(refinement, [10, 15, 5, 20, 0, 15, 15, 5])
        np.testing.assert_equal(refresh, [10, 5, -5, 0, 0, -10, -15, 10])
        text = " ".join(item.get_text() for item in figure.findobj(Text))
        self.assertIn("7 positive · 1 tie", text)
        self.assertIn("3 positive · 3 negative · 2 ties", text)
        self.assertIn("24 wins / 7 losses", text)
        self.assertIn("no task-level intervals", text)

    def test_timing_figure_normalizes_each_coordinate_to_native(self):
        import numpy as np

        visuals.style()
        with patch.object(visuals, "save_figure") as save:
            visuals.arithmetic_timing_figure(self.data, Path("unused.pdf"))
        axis = save.call_args.args[0].axes[0]
        arithmetic, timing = axis.lines[:2]
        self.assertEqual(arithmetic.get_xdata().tolist(), [2, 3, 5, 8, 10, 20])
        self.assertEqual(timing.get_xdata().tolist(), [2, 3, 5, 8, 10, 20])
        self.assertAlmostEqual(arithmetic.get_ydata()[4], 1.0)
        self.assertAlmostEqual(timing.get_ydata()[4], 1.0)
        self.assertAlmostEqual(1 - arithmetic.get_ydata()[2], 0.0893, places=3)
        self.assertAlmostEqual(1 - timing.get_ydata()[2], 0.4055, places=3)

    def test_study_design_figure_keeps_selection_and_confirmation_separate(self):
        from matplotlib.text import Text

        visuals.style()
        with patch.object(visuals, "save_figure") as save:
            visuals.study_design_figure(self.data, Path("unused.pdf"))
        text = " ".join(
            item.get_text() for item in save.call_args.args[0].findobj(Text)
        )
        self.assertIn("480 attempts · 12 settings", text)
        self.assertIn("3 denoising comparisons", text)
        self.assertIn("1,120 attempts · 7 settings", text)
        self.assertIn("later-amended descriptive analyses", text)

    def test_undersized_figure_text_is_rejected(self):
        import matplotlib.pyplot as plt

        visuals.style()
        figure = plt.figure()
        try:
            figure.text(0.5, 0.5, "Too small at print size", fontsize=8)
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "smaller than 8.5 pt"):
                    visuals.save_figure(figure, Path(directory) / "figure.pdf")
        finally:
            plt.close(figure)

    def test_clipped_figure_text_is_rejected(self):
        import matplotlib.pyplot as plt

        visuals.style()
        figure = plt.figure()
        try:
            figure.text(-0.1, 0.5, "Outside canvas", fontsize=10)
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "Clipped figure text"):
                    visuals.save_figure(figure, Path(directory) / "figure.pdf")
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
