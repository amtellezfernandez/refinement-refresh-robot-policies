#!/usr/bin/env python3
"""Run offline tests and replay the mutation experiment; no model/GPU rollouts."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for directory in (ROOT, ROOT / "experiments", ROOT / "experiments/real_smolvla",
                  ROOT / "experiments/instrument_validation", ROOT / "experiments/execution"):
    sys.path.insert(0, str(directory))


def main():
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        "test_trace", "test_evaluation_plan", "test_ledger", "test_result_metrics",
        "test_experiment", "test_run_paired", "test_analysis",
    ])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    from audit_archive import ROOT as ARCHIVE_ROOT, audit, markdown, solver_proxy_table
    report = audit()
    expected = {
        "archive_audit.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
        "ARCHIVE_AUDIT.md": markdown(report),
        "solver_proxy_table.tex": solver_proxy_table(report),
    }
    for name, content in expected.items():
        if (ARCHIVE_ROOT / name).read_text(encoding="utf-8") != content:
            raise ValueError(f"Stale generated archive audit: {name}")
    import run_experiment as mutation
    mutation_report = mutation.execute(mutation.load_protocol())
    mutation_summary = mutation.summarize(mutation_report)
    mutation.write_or_check({
        "trials.json": json.dumps(mutation_report, indent=2, allow_nan=False) + "\n",
        "summary.json": json.dumps(mutation_summary, indent=2, allow_nan=False) + "\n",
        "mutation_results_table.tex": mutation.latex_table(mutation_summary),
    }, check=True)
    print(f"OFFLINE CHECKS: {result.testsRun} tests; {len(result.skipped)} skipped; "
          f"{report['artifact_count']} archived results reconcile; "
          f"{mutation_report['trial_count']} constructed mutation trials / "
          f"{mutation_report['layer_checks']} layer checks reproduced. No new model inference or rollouts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
