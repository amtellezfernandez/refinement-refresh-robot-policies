#!/usr/bin/env python3
"""Protocol B analysis (02 B6; decisions D1, D2, D6). Python 3.8+, stdlib only. One model at a time, two-sided.

Input: the model's schedule document and the runner's final-class summary (runner_v2.py). Nothing here reads
outcomes unless every one of the 800 planned attempts has a final class and the run did not stop; a stopped run
gets its verdict from accounting alone.

Primary (D1): with all 400 pairs complete, the exact cohort contrast (n10 - n01) / 400 and the four counts.
Bounds (D2): worst-case imputation over all 400 identities (denominator 400). Direction only if the bounds
exclude zero. The complete-pair estimate is labelled separately.
Secondary (D1): fresh states on the same 40 tasks, equal task weights, seeds averaged within state, within-task
variance; reported only with complete data and a passed, hash-bound R3 coverage record (02 B6.3-B6.4).
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify as C  # noqa: E402
from v3common import CONDITIONS, CONFIRMATION_POPULATION, atomic_write_json, file_sha, sha  # noqa: E402

MAX_INCOMPLETE_PAIRS = 20          # D2
MAX_ARM_MISSING_DIFFERENCE = 10    # D2, after permitted retries
SECONDARY_LEVEL = 0.95
SECONDARY_DF_RULE = "sum over tasks of (state slots - 1)"  # 02 B6.3, PROPOSED: 40 x (5 - 1) = 160
RUN_STATUSES = ("COMPLETED", "STOPPED_INSTRUMENT_INCIDENT", "STOPPED_PROTOCOL", "HALTED_INFRA_POST",
                "HALTED_GATE_OR_HASH")


# ---------- Student t quantile (stdlib only) ----------

def _betacf(a, b, x):
    """Continued fraction for the regularized incomplete beta (modified Lentz)."""
    tiny, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise ArithmeticError("incomplete beta did not converge")


def _betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    x = df / (df + t * t)
    tail = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_quantile(p, df):
    """Inverse of t_cdf by bisection; |error| < 1e-12."""
    if not 0.0 < p < 1.0 or df <= 0:
        raise ValueError("bad quantile arguments")
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return (lo + hi) / 2.0


# ---------- pairing and accounting ----------

def pair_table(schedule, finals, population=CONFIRMATION_POPULATION):
    """identity_id -> {"suite", "task_id", "state_slot", "policy_seed", "R22": final, "R25": final}."""
    rows = schedule["schedule"]
    if sha(rows) != schedule.get("schedule_sha256"):
        raise ValueError("schedule row hash mismatch")
    expected_rows = 2 * population.identity_count()
    if len(rows) != expected_rows:
        raise ValueError("schedule must have {} rows".format(expected_rows))
    pairs = {}
    for row in rows:
        planned = row["planned_run_id"]
        if planned not in finals:
            raise ValueError("no final class for {}: the analysis never fills a gap silently".format(planned))
        final = finals[planned]
        if final not in C.FINAL_SCIENTIFIC + (C.MISSING, C.INVALIDATED):
            raise ValueError("{} has non-final class {!r}".format(planned, final))
        entry = pairs.setdefault(row["identity_id"], dict(row["episode_identity"]))
        if row["condition_id"] in entry:
            raise ValueError("duplicate condition for {}".format(row["identity_id"]))
        entry[row["condition_id"]] = final
    if len(pairs) != population.identity_count() or any(set(CONDITIONS) - set(p) for p in pairs.values()):
        raise ValueError("pairs incomplete in the schedule")
    if len(finals) != expected_rows:
        raise ValueError("final classes exist for attempts outside the schedule")
    return pairs


def missingness(pairs, population=CONFIRMATION_POPULATION):
    """Missing attempts by arm and suite, and incomplete pairs (D2, always reported)."""
    table = {suite: {"R22_missing": 0, "R25_missing": 0, "incomplete_pairs": 0} for suite in population.suites}
    total = {"R22_missing": 0, "R25_missing": 0, "incomplete_pairs": 0}
    for entry in pairs.values():
        incomplete = False
        for arm in CONDITIONS:
            if C.outcome_value(entry[arm]) is None:
                table[entry["suite"]][arm + "_missing"] += 1
                total[arm + "_missing"] += 1
                incomplete = True
        if incomplete:
            table[entry["suite"]]["incomplete_pairs"] += 1
            total["incomplete_pairs"] += 1
    total["arm_missing_difference"] = abs(total["R22_missing"] - total["R25_missing"])
    breached = []
    if total["incomplete_pairs"] > MAX_INCOMPLETE_PAIRS:
        breached.append("incomplete_pairs {} > {}".format(total["incomplete_pairs"], MAX_INCOMPLETE_PAIRS))
    if total["arm_missing_difference"] > MAX_ARM_MISSING_DIFFERENCE:
        breached.append("arm_missing_difference {} > {}".format(total["arm_missing_difference"],
                                                                MAX_ARM_MISSING_DIFFERENCE))
    return {"overall": total, "by_suite": table, "thresholds_breached": breached}


def paired_counts(pairs, population=CONFIRMATION_POPULATION):
    """The four paired outcome counts over complete pairs, overall and per suite."""
    def empty():
        return {"n11": 0, "n10": 0, "n01": 0, "n00": 0}
    overall, by_suite = empty(), {suite: empty() for suite in population.suites}
    for entry in pairs.values():
        y22, y25 = C.outcome_value(entry["R22"]), C.outcome_value(entry["R25"])
        if y22 is None or y25 is None:
            continue
        cell = "n{}{}".format(y22, y25)
        overall[cell] += 1
        by_suite[entry["suite"]][cell] += 1
    return overall, by_suite


def bounds(pairs):
    """Worst-case bounds over ALL identities; denominator is the identity count, never N_c (D2)."""
    low = high = 0
    for entry in pairs.values():
        y22, y25 = C.outcome_value(entry["R22"]), C.outcome_value(entry["R25"])
        low += (0 if y22 is None else y22) - (1 if y25 is None else y25)
        high += (1 if y22 is None else y22) - (0 if y25 is None else y25)
    n = len(pairs)
    return low / float(n), high / float(n)


def verdict(stop, incident, breached, lower, upper):
    """02 B6.5 verdict table; every combination is mapped."""
    if stop:
        return "UNSUPPORTED_REFINEMENT_SETTING"
    if incident or breached:
        return "UNINTERPRETABLE"
    if lower > 0:
        return "DELTA_POSITIVE"
    if upper < 0:
        return "DELTA_NEGATIVE"
    return "DELTA_INCONCLUSIVE"


# ---------- secondary interval (02 B6.3) ----------

def secondary_interval(pairs, population=CONFIRMATION_POPULATION, level=SECONDARY_LEVEL):
    """Fresh states on the same tasks. Requires every pair complete; raises otherwise."""
    slots, seeds = population.state_slots, population.policy_seeds
    index = {(p["suite"], p["task_id"], p["state_slot"], p["policy_seed"]): p for p in pairs.values()}
    if len(index) != len(pairs):
        raise ValueError("duplicate identity in pairs")
    task_means, variances = [], []
    for suite in population.suites:
        for task_id in population.task_ids:
            state_values = []
            for slot in slots:
                total = 0.0
                for seed in seeds:
                    entry = index.get((suite, task_id, slot, seed))
                    if entry is None:
                        raise ValueError("identity missing from pairs")
                    y22, y25 = C.outcome_value(entry["R22"]), C.outcome_value(entry["R25"])
                    if y22 is None or y25 is None:
                        raise ValueError("secondary interval needs complete data")
                    total += y22 - y25
                state_values.append(total / len(seeds))
            mean = sum(state_values) / len(state_values)
            task_means.append(mean)
            variances.append(sum((v - mean) ** 2 for v in state_values) / (len(state_values) - 1))
    return stratified_interval(task_means, variances, len(slots), level)


def stratified_interval(task_means, variances, states_per_task, level=SECONDARY_LEVEL):
    """Equal-weight stratified mean with within-task variance; t reference with df = T x (S - 1)."""
    tasks = len(task_means)
    estimate = sum(task_means) / tasks
    se = math.sqrt(sum(v / states_per_task for v in variances)) / tasks
    df = tasks * (states_per_task - 1)
    critical = t_quantile(0.5 + level / 2.0, df)
    return {"estimate": estimate, "se": se, "df": df, "critical_value": critical,
            "lower": estimate - critical * se, "upper": estimate + critical * se, "level": level,
            "target": "expected R22 - R25 difference for fresh states on the same tasks, equal task weights"}


def check_r3_record(path):
    """The R3 record must have passed and be bound to this exact analysis code and grid."""
    import r3_coverage
    record = json.loads(Path(path).read_text("utf-8"))
    if record.get("schema") != r3_coverage.SCHEMA:
        return False, "unexpected R3 record schema"
    if record.get("analysis_sha256") != file_sha(__file__):
        return False, "R3 record was produced for different analysis code"
    if record.get("r3_code_sha256") != file_sha(r3_coverage.__file__):
        return False, "R3 record was produced for different coverage code"
    if record.get("grid_sha256") != r3_coverage.grid_sha256():
        return False, "R3 record grid differs from the preregistered grid"
    if record.get("criterion") != r3_coverage.CRITERION or record.get("replicates") != r3_coverage.REPLICATES:
        return False, "R3 record criterion or replicate count differs"
    if record.get("status") != "PASS":
        return False, "R3 coverage check did not pass"
    if not record.get("authorisation_sha256"):
        return False, "R3 record is not bound to an authorisation"
    # Recompute the pass from the cells; never trust the record's own status alone.
    expected = r3_coverage.cells()
    cells = record.get("cells", [])
    if [{k: c.get(k) for k in ("baseline", "discordance", "delta", "rho_task", "rho_state", "feasible")} for c in cells] != expected:
        return False, "R3 record cells differ from the preregistered grid"
    for cell in cells:
        if cell["feasible"] and not (isinstance(cell.get("coverage"), float) and r3_coverage.CRITERION <= cell["coverage"] <= 1.0):
            return False, "R3 record has a feasible cell outside the passing coverage range"
    return True, "R3 coverage check passed"


# ---------- top level ----------

def check_journal(summary, journal_dir):
    """The summary's journal checksums must match the journal on disk (finding: summary not trusted alone)."""
    from journal import AttemptJournal
    if not Path(journal_dir).is_dir():
        raise ValueError("attempt journal directory does not exist")
    journal = AttemptJournal(journal_dir)
    if journal.checksums() != summary.get("journal_checksums"):
        raise ValueError("runner summary does not match the attempt journal on disk")
    return journal


def analyze(schedule, summary, r3_record_path=None, population=CONFIRMATION_POPULATION, journal_dir=None):
    model = schedule["model"]
    journal = check_journal(summary, journal_dir) if journal_dir is not None else None
    if summary.get("model") != model or summary.get("schedule_sha256") != schedule["schedule_sha256"]:
        raise ValueError("runner summary is for a different model or schedule")
    status = summary.get("run_status")
    if status not in RUN_STATUSES:
        raise ValueError("unknown run_status {!r}".format(status))
    report = {"model": model, "schedule_sha256": schedule["schedule_sha256"], "run_status": status,
              "two_sided": True, "prediction": "NONE_RECORDED"}
    if status != "COMPLETED":
        # 02 B9: outcomes of a stopped run are not read. Accounting only.
        report["verdict"] = "UNSUPPORTED_REFINEMENT_SETTING" if status == "STOPPED_PROTOCOL" else "UNINTERPRETABLE"
        report["accounting"] = summary.get("accounting")
        report["outcomes_read"] = False
        return report
    finals = summary["final_classes"]
    if journal is not None:
        from runner_v2 import classes_by_row

        journal_finals = {planned: C.final_class(classes)
                          for planned, classes in classes_by_row(journal).items()}
        if finals != journal_finals:
            raise ValueError("runner summary final classes do not match the attempt journal")
    pairs = pair_table(schedule, finals, population)
    miss = missingness(pairs, population)
    overall, by_suite = paired_counts(pairs, population)
    n_complete = sum(overall.values())
    lower, upper = bounds(pairs)
    report.update({"outcomes_read": True, "missingness": miss, "paired_counts": overall,
                   "paired_counts_by_suite": by_suite, "complete_pairs": n_complete,
                   "worst_case_bounds": {"lower": lower, "upper": upper, "denominator": len(pairs)}})
    if n_complete == len(pairs):
        report["exact_cohort_contrast"] = (overall["n10"] - overall["n01"]) / float(len(pairs))
    else:
        report["exact_cohort_contrast"] = None
    report["complete_pair_estimate"] = ((overall["n10"] - overall["n01"]) / float(n_complete)
                                        if n_complete else None)
    report["complete_pair_estimate_label"] = "complete pairs only; not the cohort contrast unless N_c = 400"
    report["verdict"] = verdict(False, False, bool(miss["thresholds_breached"]), lower, upper)
    # Secondary: complete data AND passed R3; never changes the verdict.
    if n_complete != len(pairs):
        report["secondary"] = {"status": "NOT_REPORTED_INCOMPLETE_DATA"}
    elif r3_record_path is None:
        report["secondary"] = {"status": "NOT_REPORTED_R3_NOT_PASSED", "reason": "no R3 record supplied"}
    else:
        passed, reason = check_r3_record(r3_record_path)
        if passed:
            report["secondary"] = dict(secondary_interval(pairs, population), status="REPORTED",
                                       r3_record_sha256=file_sha(r3_record_path))
        else:
            report["secondary"] = {"status": "NOT_REPORTED_R3_NOT_PASSED", "reason": reason}
    return report


def main():
    parser = argparse.ArgumentParser(description="Protocol B analysis for one model (02 B6).")
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True, help="runner_v2.py run_summary.json")
    parser.add_argument("--r3-record", type=Path)
    parser.add_argument("--journal", type=Path, required=True, help="the run's attempts/ directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("REFUSED: output exists")
    schedule = json.loads(args.schedule.read_text("utf-8"))
    summary = json.loads(args.summary.read_text("utf-8"))
    report = analyze(schedule, summary, args.r3_record, journal_dir=args.journal)
    report["analysis_sha256"] = file_sha(__file__)
    atomic_write_json(args.output, report)
    print(json.dumps({"model": report["model"], "verdict": report["verdict"]}))


if __name__ == "__main__":
    main()
