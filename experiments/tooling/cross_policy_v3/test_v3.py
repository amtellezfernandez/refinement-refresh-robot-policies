"""Offline unit tests for the v3 tooling. Stdlib only. No simulator, model, GPU, fixture or reserved identity.

Schedules use a synthetic test population (cohort "v3_unit_test", seeds 990001/990002), never the confirmation
cohort. The analysis is exercised on fabricated final classes. Nothing here is evidence of qualification.
"""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis_v2 as AN  # noqa: E402
import attestation as A  # noqa: E402
import classify as C  # noqa: E402
import evaluator_validation as EV  # noqa: E402
import make_schedule_v2 as MS  # noqa: E402
import qualification as Q  # noqa: E402
import runner_v2 as R  # noqa: E402
from journal import AttemptJournal  # noqa: E402
from v3common import CONFIRMATION_SEEDS, Population, SUITES, TASK_IDS, STATE_SLOTS, sha, state_key  # noqa: E402

TEST_POP = Population("v3_unit_test", SUITES, TASK_IDS, STATE_SLOTS, (990001, 990002), reserved=False)


def h(*parts):
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def manifest(model="pi05", population=TEST_POP):
    states = [{"key": state_key(s, t, k), "suite": s, "task_id": t, "state_slot": k, "draw": 0, "seed": 1,
               "state_length": 79, "state_sha256": h(model, s, t, k), "task_bddl_sha256": h("bddl", s, t),
               "task_language": "task {} {}".format(s, t)}
              for s in population.suites for t in population.task_ids for k in population.state_slots]
    return {"schema_version": 1, "cohort": population.cohort, "model": model, "state_count": len(states),
            "states": states, "states_sha256": sha(states), "initial_states_npz_sha256": h("npz"),
            "generator_sha256": h("gen"), "versions": {"mujoco": "x"}}


class Schedule(unittest.TestCase):
    def test_800_rows_paired_balanced(self):
        doc = MS.build_schedule(manifest(), "pi05", TEST_POP)
        rows = doc["schedule"]
        self.assertEqual(len(rows), 800)
        self.assertEqual(sum(r["condition_id"] == "R22" for r in rows), 400)
        self.assertEqual(doc["schedule_sha256"], sha(rows))
        for i in range(0, 800, 2):
            self.assertEqual(rows[i]["identity_id"], rows[i + 1]["identity_id"])
            self.assertEqual(rows[i]["state_sha256"], rows[i + 1]["state_sha256"])
        self.assertTrue(all(r["episode_identity"]["policy_seed"] not in CONFIRMATION_SEEDS for r in rows))
        self.assertEqual({(r["d"], r["h"]) for r in rows if r["condition_id"] == "R25"}, {(5, 2)})

    def test_deterministic(self):
        self.assertEqual(MS.build_schedule(manifest(), "pi05", TEST_POP)["schedule_sha256"],
                         MS.build_schedule(manifest(), "pi05", TEST_POP)["schedule_sha256"])

    def test_models_get_different_orders(self):
        a = MS.build_schedule(manifest("pi05"), "pi05", TEST_POP)["schedule"]
        b = MS.build_schedule(manifest("smolvla"), "smolvla", TEST_POP)["schedule"]
        self.assertNotEqual([r["identity_id"] for r in a], [r["identity_id"] for r in b])

    def test_fails_closed(self):
        cases = []
        m = manifest(); m["model"] = "smolvla"; cases.append(m)  # noqa: E702
        m = manifest(); m["states"] = m["states"][:-1]; cases.append(m)  # noqa: E702
        m = manifest(); m["states"][3]["state_sha256"] = m["states"][4]["state_sha256"]; m["states_sha256"] = sha(m["states"]); cases.append(m)  # noqa: E702,E501
        m = manifest(); m["states"][0]["state_sha256"] = "zz"; m["states_sha256"] = sha(m["states"]); cases.append(m)  # noqa: E702,E501
        m = manifest(); m["states_sha256"] = h("wrong"); cases.append(m)  # noqa: E702
        m = manifest(); m["cohort"] = "cross_policy_confirmation_v2"; cases.append(m)  # noqa: E702
        m = manifest(); del m["versions"]; cases.append(m)  # noqa: E702
        m = manifest(); m["states"][7]["task_language"] = "other"; m["states_sha256"] = sha(m["states"]); cases.append(m)  # noqa: E702,E501
        for bad in cases:
            with self.assertRaises(MS.ManifestError):
                MS.build_schedule(bad, "pi05", TEST_POP)

    def test_disjointness(self):
        with self.assertRaises(MS.ManifestError):
            MS.build_schedule(manifest(), "pi05", TEST_POP, exclude=[manifest()])


class Classes(unittest.TestCase):
    def test_final_class(self):
        self.assertEqual(C.final_class([C.SCI_SUCCESS]), C.SCI_SUCCESS)
        self.assertIsNone(C.final_class([C.INFRA_PRE]))
        self.assertEqual(C.final_class([C.INFRA_PRE, C.SCI_FAILURE]), C.SCI_FAILURE)
        self.assertEqual(C.final_class([C.INFRA_PRE, C.INFRA_PRE]), C.MISSING)
        self.assertEqual(C.final_class([C.INFRA_POST]), C.MISSING)
        self.assertEqual(C.final_class([C.SCI_SUCCESS, C.SCI_SUCCESS]), C.INVALIDATED)
        self.assertEqual(C.final_class([C.INFRA_POST, C.SCI_SUCCESS]), C.INVALIDATED)

    def test_marker_decides_pre_post(self):
        self.assertEqual(C.classify_exception(RuntimeError("x"), False), C.INFRA_PRE)
        self.assertEqual(C.classify_exception(RuntimeError("x"), True), C.INFRA_POST)
        self.assertEqual(C.classify_exception(C.InfraPre("x"), True), C.INFRA_POST)
        self.assertEqual(C.classify_exception(C.InstrumentIncident("x"), True), C.INSTRUMENT_INCIDENT)

    def test_outcome(self):
        self.assertEqual(C.classify_outcome({"success": True, "termination_reason": "success"}), C.SCI_SUCCESS)
        self.assertEqual(C.classify_outcome({"success": False, "termination_reason": "nonfinite_action_valid_inputs"}),
                         C.SCI_FAILURE)
        with self.assertRaises(ValueError):
            C.classify_outcome({"success": True, "termination_reason": "task_horizon"})


class Attestation(unittest.TestCase):
    def good(self, verifier_gate="g" * 64, d=5, tick=4, seq=1):
        rid = A.request_id("row", 0, tick)
        return rid, A.build(rid, tick, d, d, A.expected_time_values(d), seq, verifier_gate, "F")

    def test_time_grid(self):
        self.assertEqual(A.expected_time_values(2), [1.0, 0.5])
        self.assertEqual(len(A.expected_time_values(10)), 10)

    def test_accepts_correct(self):
        rid, att = self.good()
        A.Verifier("g" * 64, "F").verify(att, rid, 4, 5)

    def test_refuses_each_failure(self):
        rid, att = self.good()
        mutations = [
            lambda a: None,
            lambda a: dict(a, executed_head_evaluations=10),
            lambda a: dict(a, tick=5),
            lambda a: dict(a, request_id=A.request_id("row", 0, 5)),
            lambda a: dict(a, d_requested=2),
            lambda a: dict(a, time_values=a["time_values"][:-1]),
            lambda a: dict(a, startup_gate_sha256="h" * 64),
            lambda a: dict(a, xla_flags="other"),
        ]
        for mutate in mutations:
            bad = mutate(dict(att))
            if isinstance(bad, dict) and "attestation_sha256" in bad and mutate is not mutations[0]:
                bad["attestation_sha256"] = A.attestation_digest(bad)  # a consistent forgery still fails
            with self.assertRaises(C.InstrumentIncident):
                A.Verifier("g" * 64, "F").verify(bad, rid, 4, 5)
        tampered = dict(att, executed_head_evaluations=5)
        tampered["sequence_number"] = 2
        with self.assertRaises(C.InstrumentIncident):  # digest no longer matches
            A.Verifier("g" * 64, "F").verify(tampered, rid, 4, 5)

    def test_sequence_must_increase(self):
        verifier = A.Verifier("g" * 64, "F")
        rid, att = self.good(seq=3)
        verifier.verify(att, rid, 4, 5)
        with self.assertRaises(C.InstrumentIncident):
            verifier.verify(att, rid, 4, 5)

    def test_no_gate_record_refused(self):
        with self.assertRaises(C.InstrumentIncident):
            A.Verifier(None, "F")


class FakeAdapter(object):
    """Scripted attempt outcomes keyed by (planned_run_id, attempt_number). No environment exists."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def stack_config(self):
        return {"fake": True}

    def provenance_sha256(self):
        return "p" * 64

    def run_attempt(self, context):
        key = (context.row["planned_run_id"], context.attempt_number)
        self.calls.append(key)
        action = self.script.get(key, "success")
        if action == "pre":
            raise RuntimeError("env build failed")
        context.record_reset("s", "o")
        context.mark_first_query()
        if action == "post":
            raise ConnectionError("websocket lost")
        if action == "incident":
            raise C.InstrumentIncident("attestation check 4 failed")
        if action == "malformed":
            return {"outcome": {"success": True, "termination_reason": "task_horizon"},
                    "settled_state_sha256": "s", "first_observation_sha256": "o", "action_provenance": []}
        success = action == "success"
        return {"outcome": {"success": success, "termination_reason": "success" if success else "task_horizon"},
                "settled_state_sha256": "s", "first_observation_sha256": "o", "action_provenance": []}


class Runner(unittest.TestCase):
    def setUp(self):
        self.rows = MS.build_schedule(manifest(), "pi05", TEST_POP)["schedule"][:6]
        self.dir = Path(tempfile.mkdtemp())

    def run_until_done(self, adapter, launches=5):
        journal = AttemptJournal(self.dir / "attempts")
        for _ in range(launches):
            status, code = R.execute(self.rows, journal, adapter, self.dir, "p" * 64, ["test"])
            if code != R.EXIT_FRESH_PROCESS_RETRY:
                return status, code, journal
        raise AssertionError("did not finish")

    def test_completes_and_classifies(self):
        script = {(self.rows[1]["planned_run_id"], 0): "failure"}
        status, code, journal = self.run_until_done(FakeAdapter(script))
        self.assertEqual((status, code), ("COMPLETED", R.EXIT_COMPLETED))
        finals = {k: C.final_class(v) for k, v in R.classes_by_row(journal).items()}
        self.assertEqual(finals[self.rows[1]["planned_run_id"]], C.SCI_FAILURE)
        self.assertEqual(len(finals), 6)

    def test_infra_pre_retried_once_in_fresh_process(self):
        planned = self.rows[2]["planned_run_id"]
        adapter = FakeAdapter({(planned, 0): "pre"})
        journal = AttemptJournal(self.dir / "attempts")
        status, code = R.execute(self.rows, journal, adapter, self.dir, "p" * 64, ["t"])
        self.assertEqual(code, R.EXIT_FRESH_PROCESS_RETRY)  # retry refused in the same process
        status, code = R.execute(self.rows, journal, adapter, self.dir, "p" * 64, ["t"])
        self.assertEqual(status, "COMPLETED")
        self.assertEqual(adapter.calls[3], (planned, 1))  # first call of the new process is the retry
        self.assertEqual(R.classes_by_row(journal)[planned], [C.INFRA_PRE, C.SCI_SUCCESS])

    def test_infra_post_not_retried(self):
        planned = self.rows[0]["planned_run_id"]
        status, code, journal = self.run_until_done(FakeAdapter({(planned, 0): "post"}))
        self.assertEqual(status, "COMPLETED")
        self.assertEqual(R.classes_by_row(journal)[planned], [C.INFRA_POST])
        self.assertEqual(C.final_class([C.INFRA_POST]), C.MISSING)

    def test_incident_stops_and_never_resumes(self):
        planned = self.rows[3]["planned_run_id"]
        adapter = FakeAdapter({(planned, 0): "incident"})
        journal = AttemptJournal(self.dir / "attempts")
        status, code = R.execute(self.rows, journal, adapter, self.dir, "p" * 64, ["t"])
        self.assertEqual((status, code), ("STOPPED_INSTRUMENT_INCIDENT", R.EXIT_STOPPED))
        self.assertTrue((self.dir / "incidents").is_dir())
        calls = len(adapter.calls)
        status, code = R.execute(self.rows, journal, adapter, self.dir, "p" * 64, ["t"])
        self.assertEqual(code, R.EXIT_STOPPED)
        self.assertEqual(len(adapter.calls), calls)  # nothing ran after the stop
        summary = R.summarize(self.rows, journal, self.dir, "x", "pi05", status)
        self.assertEqual(len(summary["accounting"]["not_executed"]), 2)

    def test_malformed_result_is_incident(self):
        planned = self.rows[0]["planned_run_id"]
        status, code, journal = self.run_until_done(FakeAdapter({(planned, 0): "malformed"}))
        self.assertEqual(status, "STOPPED_INSTRUMENT_INCIDENT")

    def test_accounting_by_arm_and_suite(self):
        status, code, journal = self.run_until_done(FakeAdapter({}))
        summary = R.summarize(self.rows, journal, self.dir, "x", "pi05", status)
        cells = summary["accounting"]["by_arm_and_suite"]
        self.assertEqual(sum(c[C.SCI_SUCCESS] for c in cells.values()), 6)

    def test_interrupted_after_first_query_is_post(self):
        journal = AttemptJournal(self.dir / "attempts")
        row = self.rows[0]
        path = journal.begin(row, "p", ["t"])
        journal.update(path, "running")
        Path(str(path) + ".first_query").write_text("{}")
        status, code = R.execute(self.rows, journal, FakeAdapter({}), self.dir, "p" * 64, ["t"])
        self.assertEqual(R.classes_by_row(journal)[row["planned_run_id"]], [C.INFRA_POST])


def fake_summary(schedule, outcomes):
    finals = {}
    for row in schedule["schedule"]:
        finals[row["planned_run_id"]] = outcomes(row)
    return {"model": schedule["model"], "schedule_sha256": schedule["schedule_sha256"], "run_status": "COMPLETED",
            "final_classes": finals}


class Analysis(unittest.TestCase):
    def setUp(self):
        self.schedule = MS.build_schedule(manifest(), "pi05", TEST_POP)

    def test_completed_summary_cannot_override_journal_halt(self):
        # The last 22 missing attempts occupy 11 pairs: D2 limits alone
        # permit interpretation, but B9 halts after >20 INFRA_POST events.
        with tempfile.TemporaryDirectory() as directory:
            journal = AttemptJournal(Path(directory) / "attempts")
            rows = self.schedule["schedule"]
            for index, row in enumerate(rows):
                value = (C.INFRA_POST if index >= len(rows) - 22 else
                         C.SCI_SUCCESS if row["condition_id"] == "R22" else C.SCI_FAILURE)
                path = journal.begin(row, "synthetic", ["test"])
                journal.update(path, "running")
                journal.update(path, "failed" if value == C.INFRA_POST else "completed",
                               attempt_class=value)
            summary = R.summarize(rows, journal, directory, self.schedule["schedule_sha256"],
                                  "pi05", "HALTED_INFRA_POST")
            report = AN.analyze(self.schedule, summary, population=TEST_POP, journal_dir=journal.root)
            self.assertFalse(report["outcomes_read"])
            self.assertTrue(report["journal_checked"])
            summary["run_status"] = "COMPLETED"
            with self.assertRaisesRegex(ValueError, "halt condition"):
                AN.analyze(self.schedule, summary, population=TEST_POP, journal_dir=journal.root)

    def test_summary_final_classes_must_match_journal(self):
        summary = fake_summary(self.schedule, lambda row: C.SCI_FAILURE)
        with tempfile.TemporaryDirectory() as directory:
            journal = AttemptJournal(directory)
            for row in self.schedule["schedule"]:
                path = journal.begin(row, "synthetic", ["test"])
                journal.update(path, "running")
                journal.update(path, "completed", attempt_class=C.SCI_FAILURE)
            summary["journal_checksums"] = journal.checksums()
            report = AN.analyze(self.schedule, summary, population=TEST_POP, journal_dir=directory)
            self.assertEqual(report["paired_counts"]["n00"], 400)
            planned = self.schedule["schedule"][0]["planned_run_id"]
            summary["final_classes"][planned] = C.SCI_SUCCESS
            with self.assertRaisesRegex(ValueError, "final classes do not match"):
                AN.analyze(self.schedule, summary, population=TEST_POP, journal_dir=directory)

    def test_completed_summary_cannot_override_persisted_halt(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = AttemptJournal(Path(directory) / "attempts")
            for row in self.schedule["schedule"]:
                path = journal.begin(row, "synthetic", ["test"])
                journal.update(path, "running")
                journal.update(path, "completed", attempt_class=C.SCI_FAILURE)
            summary = R.summarize(self.schedule["schedule"], journal, directory,
                                  self.schedule["schedule_sha256"], "pi05", "COMPLETED")
            R.write_halt(directory, "HALTED_GATE_OR_HASH", "synthetic gate fault")
            with self.assertRaisesRegex(ValueError, "halt condition: HALTED_GATE_OR_HASH"):
                AN.analyze(self.schedule, summary, population=TEST_POP, journal_dir=journal.root)

    def test_analysis_rejects_rehashed_semantic_schedule_mutations(self):
        finals = fake_summary(self.schedule, lambda row: C.SCI_FAILURE)["final_classes"]
        for mutation in ("identity", "denoising", "planned_id", "state_key", "cohort"):
            schedule = json.loads(json.dumps(self.schedule))
            rows = schedule["schedule"]
            if mutation == "identity":
                for row in rows[:2]:
                    row["episode_identity"] = dict(rows[2]["episode_identity"])
            elif mutation == "denoising":
                rows[0]["d"] = 10
            elif mutation == "planned_id":
                rows[1]["planned_run_id"] = rows[0]["planned_run_id"]
            elif mutation == "state_key":
                rows[0]["state_key"] = "wrong-state"
            else:
                rows[0]["cohort"] = "wrong-cohort"
            schedule["schedule_sha256"] = sha(rows)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                AN.pair_table(schedule, finals, TEST_POP)

    def test_analyze_requires_the_journal(self):
        summary = fake_summary(self.schedule, lambda row: C.SCI_FAILURE)
        with self.assertRaisesRegex(ValueError, "journal_dir is required"):
            AN.analyze(self.schedule, summary, population=TEST_POP)
        report = AN.analyze(self.schedule, summary, population=TEST_POP, journal_dir=AN.SKIP_JOURNAL_CHECK)
        self.assertFalse(report["journal_checked"])

    def test_exact_contrast_and_counts(self):
        # R22 succeeds on task 0 of every suite only; R25 always fails.
        def outcome(row):
            ok = row["condition_id"] == "R22" and row["episode_identity"]["task_id"] == 0
            return C.SCI_SUCCESS if ok else C.SCI_FAILURE
        report = AN.analyze(self.schedule, fake_summary(self.schedule, outcome), population=TEST_POP, journal_dir=AN.SKIP_JOURNAL_CHECK)
        self.assertEqual(report["paired_counts"], {"n11": 0, "n10": 40, "n01": 0, "n00": 360})
        self.assertAlmostEqual(report["exact_cohort_contrast"], 0.1)
        self.assertEqual(report["worst_case_bounds"]["lower"], report["worst_case_bounds"]["upper"])
        self.assertEqual(report["verdict"], "DELTA_POSITIVE")
        self.assertEqual(report["secondary"]["status"], "NOT_REPORTED_R3_NOT_PASSED")

    def test_missingness_bounds_and_thresholds(self):
        missing = set()
        for row in self.schedule["schedule"]:
            if row["condition_id"] == "R25" and len(missing) < 5:
                missing.add(row["planned_run_id"])

        def outcome(row):
            if row["planned_run_id"] in missing:
                return C.MISSING
            return C.SCI_SUCCESS if row["condition_id"] == "R22" and row["episode_identity"]["task_id"] < 2 else C.SCI_FAILURE
        report = AN.analyze(self.schedule, fake_summary(self.schedule, outcome), population=TEST_POP, journal_dir=AN.SKIP_JOURNAL_CHECK)
        self.assertIsNone(report["exact_cohort_contrast"])
        self.assertEqual(report["missingness"]["overall"]["R25_missing"], 5)
        self.assertEqual(report["worst_case_bounds"]["denominator"], 400)
        self.assertEqual(report["secondary"]["status"], "NOT_REPORTED_INCOMPLETE_DATA")
        self.assertEqual(report["verdict"], "DELTA_POSITIVE")  # 5 missing cannot erase 80 discordant pairs
        many = set(r["planned_run_id"] for r in self.schedule["schedule"] if r["condition_id"] == "R25")
        many = set(sorted(many)[:11])
        report = AN.analyze(self.schedule, fake_summary(
            self.schedule, lambda r: C.MISSING if r["planned_run_id"] in many else C.SCI_FAILURE), population=TEST_POP, journal_dir=AN.SKIP_JOURNAL_CHECK)
        self.assertEqual(report["verdict"], "UNINTERPRETABLE")  # arm difference 11 > 10

    def test_bounds_containing_zero_is_inconclusive(self):
        report = AN.analyze(self.schedule, fake_summary(self.schedule, lambda r: C.SCI_FAILURE), population=TEST_POP, journal_dir=AN.SKIP_JOURNAL_CHECK)
        self.assertEqual(report["verdict"], "DELTA_INCONCLUSIVE")

    def test_stopped_run_reads_no_outcomes(self):
        summary = fake_summary(self.schedule, lambda r: C.SCI_SUCCESS)
        summary["run_status"] = "STOPPED_INSTRUMENT_INCIDENT"
        report = AN.analyze(self.schedule, summary, population=TEST_POP, journal_dir=AN.SKIP_JOURNAL_CHECK)
        self.assertFalse(report["outcomes_read"])
        self.assertEqual(report["verdict"], "UNINTERPRETABLE")

    def test_missing_final_class_refused(self):
        summary = fake_summary(self.schedule, lambda r: C.SCI_SUCCESS)
        del summary["final_classes"][self.schedule["schedule"][0]["planned_run_id"]]
        with self.assertRaises(ValueError):
            AN.analyze(self.schedule, summary, population=TEST_POP, journal_dir=AN.SKIP_JOURNAL_CHECK)

    def test_t_quantile(self):
        self.assertAlmostEqual(AN.t_quantile(0.975, 160), 1.974902, places=5)  # scipy t.ppf: 1.9749015600
        self.assertAlmostEqual(AN.t_quantile(0.975, 4), 2.776445, places=5)
        self.assertAlmostEqual(AN.t_quantile(0.975, 39), 2.022691, places=5)

    def test_secondary_interval_equals_contrast(self):
        def outcome(row):
            ok = row["condition_id"] == "R22" and row["episode_identity"]["state_slot"] == 0
            return C.SCI_SUCCESS if ok else C.SCI_FAILURE
        pairs = AN.pair_table(self.schedule, fake_summary(self.schedule, outcome)["final_classes"], TEST_POP)
        interval = AN.secondary_interval(pairs, TEST_POP)
        self.assertAlmostEqual(interval["estimate"], 0.2)
        self.assertEqual(interval["df"], 160)
        self.assertGreater(interval["se"], 0)


class Qualification(unittest.TestCase):
    def test_shipped_records_unverified_and_refused(self):
        for model in ("pi05", "smolvla"):
            statuses = Q.summary(model)
            self.assertTrue(all(s in ("UNVERIFIED", "NOT_APPLICABLE") for s in statuses.values()), statuses)
            with self.assertRaises(Q.Refused):
                Q.require_qualified(model)

    def test_changed_binding_keys_rejected(self):
        record = json.loads(Q.record_path("pi05").read_text())
        record["items"]["evaluator"]["bindings"] = {}
        with self.assertRaises(Q.RecordError):
            Q.validate(record, "pi05")

    def test_passed_needs_alba_authorisation_and_verdict(self):
        root = Path(tempfile.mkdtemp())
        (root / "auth.json").write_text(json.dumps({"approved_by": "Alba"}))
        (root / "ev.json").write_text(json.dumps({"verdict": "EVALUATOR_DISAGREES"}))
        record = json.loads(Q.record_path("pi05").read_text())
        item = record["items"]["evaluator"]
        item["status"] = "PASSED"
        item["bindings"] = {"protocol_e_record_sha256": "x", "verdict": "EVALUATOR_QUALIFIED"}
        item["evidence"] = {"run_id": "r", "authorisation_path": "auth.json",
                            "authorisation_sha256": Q.file_sha(root / "auth.json"),
                            "evidence_path": "ev.json", "evidence_sha256": Q.file_sha(root / "ev.json")}
        with self.assertRaises(Q.RecordError):  # evidence says DISAGREES
            Q.validate(record, "pi05", root=root)
        (root / "ev.json").write_text(json.dumps({"verdict": "EVALUATOR_QUALIFIED"}))
        item["evidence"]["evidence_sha256"] = Q.file_sha(root / "ev.json")
        Q.validate(record, "pi05", root=root)  # well-formed; still a synthetic record, not a qualification

    def test_passed_without_evidence_rejected(self):
        record = json.loads(Q.record_path("pi05").read_text())
        record["items"]["parity"]["status"] = "PASSED"
        with self.assertRaises(Q.RecordError):
            Q.validate(record, "pi05")


class R3Record(unittest.TestCase):
    def record(self, coverage):
        import r3_coverage as R3
        cells = [dict(c, coverage=(coverage if c["feasible"] else None)) for c in R3.cells()]
        return {"schema": R3.SCHEMA, "status": "PASS", "criterion": R3.CRITERION, "replicates": R3.REPLICATES,
                "grid_sha256": R3.grid_sha256(), "analysis_sha256": Q.file_sha(AN.__file__),
                "r3_code_sha256": Q.file_sha(R3.__file__), "authorisation_sha256": "a" * 64, "cells": cells}

    def check(self, record):
        path = Path(tempfile.mkdtemp()) / "r3.json"
        path.write_text(json.dumps(record))
        return AN.check_r3_record(path)[0]

    def test_recomputed_from_cells(self):
        self.assertTrue(self.check(self.record(0.95)))
        self.assertFalse(self.check(self.record(0.92)))  # self-declared PASS with failing cells is refused
        self.assertFalse(self.check(self.record(1.01)))
        stale_code = self.record(0.95)
        stale_code["r3_code_sha256"] = "0" * 64
        self.assertFalse(self.check(stale_code))
        unbound = self.record(0.95)
        del unbound["authorisation_sha256"]
        self.assertFalse(self.check(unbound))


class Evaluator(unittest.TestCase):
    BDDL = """(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  ; a comment (with parentheses)
  (:goal
    (And (On akita_black_bowl_1 plate_1) (Open wooden_cabinet_1_top_region))
  )
)"""

    def test_goal_parse(self):
        self.assertEqual(EV.goal_atoms(self.BDDL),
                         [["on", "akita_black_bowl_1", "plate_1"], ["open", "wooden_cabinet_1_top_region"]])
        self.assertTrue(EV.same_atoms(EV.goal_atoms(self.BDDL),
                                      [["open", "wooden_cabinet_1_top_region"], ["On", "akita_black_bowl_1", "plate_1"]]))

    def test_compare_and_verdict(self):
        reached = [{"step": i, "recorded_success": i >= 3, "independent_goal": i >= 3, "episode_ended": i == 3}
                   for i in range(4)]
        good = EV.compare(reached, "GOAL_REACHED", 10)
        self.assertTrue(good["agrees"])
        late = [dict(t, recorded_success=t["step"] >= 4) for t in reached]
        bad = EV.compare(late, "GOAL_REACHED", 10)
        self.assertFalse(bad["agrees"])
        self.assertEqual(EV.stack_verdict([good], [], None), "EVALUATOR_QUALIFIED")
        self.assertEqual(EV.stack_verdict([good, bad], [], None), "EVALUATOR_DISAGREES")
        never = EV.compare([dict(t, recorded_success=False, independent_goal=False) for t in reached], "GOAL_REACHED", 10)
        self.assertEqual(EV.stack_verdict([good, never], [], None), "EVALUATOR_NOT_QUALIFIED")
        self.assertEqual(EV.stack_verdict([good], ["libero_10/GOAL_REACHED"], None), "EVALUATOR_NOT_QUALIFIED")
        self.assertEqual(EV.stack_verdict([good], [], {"x": 1}), "UNINTERPRETABLE")

    def test_fixture_rules(self):
        base = {"schema": EV.SCHEMA, "stack": "pi05", "fixtures": [
            {"fixture_id": "f1", "suite": "libero_goal", "task_id": 0, "initial_state": {"source": "official", "index": 0},
             "action_source": {"kind": "scripted_noop", "steps": 20}, "expected_case": "GOAL_UNMET"}]}
        coverage = EV.validate_fixtures(base, "pi05")
        self.assertIn("libero_goal/GOAL_REACHED", coverage["missing_cases"])
        reserved = json.loads(json.dumps(base))
        reserved["fixtures"][0]["policy_seed"] = 810001
        with self.assertRaises(EV.FixtureError):
            EV.validate_fixtures(reserved, "pi05")
        generated = json.loads(json.dumps(base))
        generated["fixtures"][0]["initial_state"] = {"source": "generated", "index": 0}
        with self.assertRaises(EV.FixtureError):
            EV.validate_fixtures(generated, "pi05")


class ReviewDecisions(unittest.TestCase):
    def test_r3_independent_parameters(self):
        import r3_coverage as R3

        combinations = {(row["rho_task"], row["rho_state"]) for row in R3.cells()}
        self.assertEqual(combinations, {(a, b) for a in (0, .25, .5) for b in (0, .25, .5)})

    def test_benchstudio_export_refuses_unqualified_evidence(self):
        runtime = dict.fromkeys(
            ("runner", "checkpoint", "simulator_stack", "renderer", "assets", "normalization"),
            "a" * 64,
        )
        with self.assertRaisesRegex(EV.FixtureError, "unqualified"):
            EV.benchstudio_report("pi05", runtime, [], "EVALUATOR_NOT_QUALIFIED", "b" * 64)


if __name__ == "__main__":
    unittest.main()
