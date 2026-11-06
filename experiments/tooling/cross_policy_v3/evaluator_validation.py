#!/usr/bin/env python3
"""Protocol E: model-free evaluator validation for both stacks (13; decision D3). NEVER RUN.

No policy is loaded. Fixtures start from official LIBERO initial states (outside the confirmation population by
construction) and apply scripted or saved, version-matched actions. At every step the harness records:
  1. the outcome the stack itself reports (pi0.5: `done`; SmolVLA: `info["is_success"]`, terminated, truncated);
  2. the independent goal truth: the task's BDDL `:goal` parsed here from the file on disk (own parser, no
     BDDLUtils), each atom evaluated through LIBERO's atomic `eval_predicate_fn` on the env's
     `object_states_dict`, conjunction formed here. Atomic predicate geometry is shared with the evaluator and is
     in the trusted base (13 E4).
Pure functions (parser, fixture validation, comparison, verdict) run without any simulator; the stack drivers
import their stack lazily.

    evaluator_validation.py --stack pi05|smolvla --fixtures FIXTURES.json --authorisation AUTH.json --output DIR \
        [--confirmation-manifest M.json ...]
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3common import PROTOCOL_E_DOCUMENT, SUITES, atomic_write_json, file_sha, sha  # noqa: E402

SCHEMA = "protocol_e_fixtures_v1"
STACKS = ("pi05", "smolvla")
CASES = ("GOAL_UNMET", "HORIZON", "GOAL_REACHED")
NOOP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
GOAL_UNMET_STEPS = 20


class FixtureError(ValueError):
    pass


class EvaluatorIncident(RuntimeError):
    """The harness itself cannot vouch for a reading (e.g. its parse differs from the env's own)."""


# ---------- BDDL goal parser (independent of LIBERO's BDDLUtils) ----------

def tokenize(text):
    tokens, current = [], []
    for line in text.splitlines():
        line = line.split(";", 1)[0]
        for char in line:
            if char in "()":
                if current:
                    tokens.append("".join(current))
                    current = []
                tokens.append(char)
            elif char.isspace():
                if current:
                    tokens.append("".join(current))
                    current = []
            else:
                current.append(char)
        if current:
            tokens.append("".join(current))
            current = []
    return tokens


def parse_sexpr(tokens):
    stack, root = [], []
    current = root
    for token in tokens:
        if token == "(":
            new = []
            current.append(new)
            stack.append(current)
            current = new
        elif token == ")":
            if not stack:
                raise FixtureError("unbalanced ')' in BDDL")
            current = stack.pop()
        else:
            current.append(token)
    if stack:
        raise FixtureError("unbalanced '(' in BDDL")
    return root


def find_goal(tree):
    for node in tree:
        if isinstance(node, list):
            if node and node[0] == ":goal":
                return node
            found = find_goal(node)
            if found is not None:
                return found
    return None


def goal_atoms(bddl_text):
    """The :goal section as [[predicate, obj, ...], ...], predicate lower-cased. Conjunction only."""
    goal = find_goal(parse_sexpr(tokenize(bddl_text)))
    if goal is None or len(goal) != 2 or not isinstance(goal[1], list):
        raise FixtureError("no single :goal expression")
    expression = goal[1]
    if expression and isinstance(expression[0], str) and expression[0].lower() == "and":
        atoms = expression[1:]
    else:
        atoms = [expression]
    out = []
    for atom in atoms:
        if not isinstance(atom, list) or len(atom) not in (2, 3) or not all(isinstance(a, str) for a in atom):
            raise FixtureError("unsupported goal atom {!r}: only unary/binary predicates in a conjunction".format(atom))
        out.append([atom[0].lower()] + atom[1:])
    return out


def same_atoms(ours, env_goal_state):
    """Cross-check against the env's own parsed_problem['goal_state'] (13 E4)."""
    theirs = [[str(a[0]).lower()] + [str(x) for x in a[1:]] for a in env_goal_state]
    return sorted(map(tuple, ours)) == sorted(map(tuple, theirs))


def independent_goal(problem, atoms):
    """Each atom via LIBERO's atomic predicate; the conjunction here. Returns (all_true, per-atom list)."""
    from libero.libero.envs.predicates import eval_predicate_fn
    values = []
    for atom in atoms:
        objects = [problem.object_states_dict[name] for name in atom[1:]]
        values.append(bool(eval_predicate_fn(atom[0], *objects)))
    return all(values), values


# ---------- fixtures ----------

def validate_fixtures(document, stack, root=Path(".")):
    if document.get("schema") != SCHEMA or document.get("stack") != stack:
        raise FixtureError("wrong fixture schema or stack")
    fixtures = document.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise FixtureError("no fixtures")
    seen, coverage = set(), {(s, c): 0 for s in SUITES for c in CASES}
    for fixture in fixtures:
        for field in ("fixture_id", "suite", "task_id", "initial_state", "action_source", "expected_case"):
            if field not in fixture:
                raise FixtureError("fixture lacks {}".format(field))
        if fixture["fixture_id"] in seen:
            raise FixtureError("duplicate fixture_id")
        seen.add(fixture["fixture_id"])
        if fixture["suite"] not in SUITES or fixture["expected_case"] not in CASES:
            raise FixtureError("bad suite or case in {}".format(fixture["fixture_id"]))
        initial = fixture["initial_state"]
        # Outside the confirmation population: official initial states only; no policy seed of any kind.
        if initial.get("source") != "official" or not isinstance(initial.get("index"), int):
            raise FixtureError("fixture {} must start from an official initial state".format(fixture["fixture_id"]))
        if "policy_seed" in fixture or "state_slot" in fixture:
            raise FixtureError("fixture {} names a policy seed or state slot".format(fixture["fixture_id"]))
        source = fixture["action_source"]
        kind = source.get("kind")
        if kind == "scripted_noop":
            if fixture["expected_case"] == "GOAL_REACHED":
                raise FixtureError("a no-op sequence cannot be a goal-reached fixture")
            steps = source.get("steps")
            if fixture["expected_case"] == "HORIZON" and steps != "horizon":
                raise FixtureError("HORIZON fixtures run no-ops to the stack's horizon")
            if fixture["expected_case"] == "GOAL_UNMET" and steps != GOAL_UNMET_STEPS:
                raise FixtureError("GOAL_UNMET fixtures run {} no-op steps".format(GOAL_UNMET_STEPS))
        elif kind in ("saved", "scripted_actions"):
            path = (Path(root) / source.get("path", "")).resolve()
            if Path(root).resolve() not in path.parents or not path.is_file() or file_sha(path) != source.get("sha256"):
                raise FixtureError("saved actions missing or hash mismatch for {}".format(fixture["fixture_id"]))
            if not source.get("version_matched_to"):
                raise FixtureError("action fixtures must name the native stack version (D3)")
            if kind == "scripted_actions":
                script = (Path(root) / source.get("script_path", "")).resolve()
                if (Path(root).resolve() not in script.parents or not script.is_file()
                        or file_sha(script) != source.get("script_sha256")):
                    raise FixtureError("scripted actions require a frozen script source")
        else:
            raise FixtureError("unknown action source {!r}".format(kind))
        coverage[(fixture["suite"], fixture["expected_case"])] += 1
    missing = sorted("{}/{}".format(s, c) for (s, c), n in coverage.items() if n == 0)
    return {"fixture_count": len(fixtures), "missing_cases": missing}


# ---------- comparison and verdict (13 E5, E6) ----------

def compare(trace, expected_case, horizon):
    """trace: [{"step", "recorded_success", "independent_goal", "episode_ended"}] in order, one per env step."""
    disagreements = [t["step"] for t in trace if bool(t["recorded_success"]) != bool(t["independent_goal"])]
    first_goal = next((t["step"] for t in trace if t["independent_goal"]), None)
    first_recorded = next((t["step"] for t in trace if t["recorded_success"]), None)
    end_step = trace[-1]["step"] if trace else None
    result = {"steps": len(trace), "disagreement_steps": disagreements, "first_goal_step": first_goal,
              "first_recorded_success_step": first_recorded, "end_step": end_step, "horizon": horizon,
              "fixture_invalid": False}
    if expected_case == "GOAL_UNMET":
        result["case_ok"] = first_goal is None and first_recorded is None and len(trace) == GOAL_UNMET_STEPS \
            and not any(t["episode_ended"] for t in trace[:-1])
    elif expected_case == "HORIZON":
        result["case_ok"] = first_goal is None and first_recorded is None and len(trace) == horizon
    else:
        if first_goal is None:
            result["fixture_invalid"], result["case_ok"] = True, False
        else:
            result["case_ok"] = first_recorded == first_goal
            result["terminates_on_success"] = bool(trace[first_goal]["episode_ended"]) if first_goal < len(trace) else None
    result["agrees"] = not disagreements and result["case_ok"]
    return result


def stack_verdict(results, missing_cases, incident):
    """13 E6 verdict table."""
    if incident:
        return "UNINTERPRETABLE"
    if any(r["disagreement_steps"] or (not r["case_ok"] and not r["fixture_invalid"]) for r in results):
        return "EVALUATOR_DISAGREES"
    if missing_cases or any(r["fixture_invalid"] for r in results):
        return "EVALUATOR_NOT_QUALIFIED"
    return "EVALUATOR_QUALIFIED"


# ---------- stack drivers (need the stack; never run) ----------

def bddl_text_and_hash(path):
    data = Path(path).read_bytes()
    return data.decode("utf-8"), hashlib.sha256(data).hexdigest()


def actions_for(fixture, horizon, root):
    import numpy as np
    source = fixture["action_source"]
    if source["kind"] == "scripted_noop":
        steps = horizon if source["steps"] == "horizon" else source["steps"]
        return [list(NOOP) for _ in range(steps)]
    actions = np.load(str(Path(root) / source["path"]), allow_pickle=False)
    if actions.ndim != 2 or actions.shape[1] != 7 or actions.dtype.kind != "f" or not np.isfinite(actions).all():
        raise FixtureError("fixture actions must be finite floating-point vectors of width 7")
    return actions.tolist()


def evaluator_sample(problem, atoms, step, native_success, stop_reason=None):
    import numpy as np

    goal, per_atom = independent_goal(problem, atoms)
    state = np.asarray(problem.sim.get_state().flatten())
    return {"step": step, "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
            "goal_met": bool(goal), "native_success": bool(native_success),
            "stop_reason": stop_reason, "per_atom": per_atom}


def benchstudio_report(stack, runtime, results, verdict, checker_sha256):
    """Translate qualified Protocol E rows into the BenchStudio launch-gate schema."""
    required = {"runner", "checkpoint", "simulator_stack", "renderer", "assets", "normalization"}
    if (
        stack not in STACKS or not isinstance(runtime, dict) or not required.issubset(runtime)
        or not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in runtime.values())
        or not isinstance(checker_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", checker_sha256)
    ):
        raise FixtureError("BenchStudio export requires the native runtime hash mapping")
    if verdict != "EVALUATOR_QUALIFIED" or not results or any(
        row.get("agrees") is not True for row in results
    ):
        raise FixtureError("cannot export unqualified Protocol E evidence")
    kinds = {"GOAL_UNMET": "goal_unmet", "HORIZON": "horizon_timeout", "GOAL_REACHED": "goal_reached"}
    cases, seen = [], set()
    for row in results:
        key = row["suite"], kinds[row["expected_case"]]
        if key in seen:
            raise FixtureError("BenchStudio export requires exactly one fixture per suite and case")
        seen.add(key)
        samples = row["benchstudio_samples"]
        if not samples or samples[0]["goal_met"] or samples[0]["native_success"]:
            raise FixtureError("BenchStudio export needs a goal-unmet reset")
        if key[1] == "goal_unmet":
            samples, outcome = samples[:1], None
        else:
            if samples[-1]["stop_reason"] is None:
                raise FixtureError("BenchStudio export needs a terminal trace or the full horizon")
            outcome = any(sample["native_success"] for sample in samples)
        cases.append({"suite": key[0], "kind": key[1], "task_id": str(row["task_id"]),
                      "horizon": row["horizon"], "goal_spec_sha256": row["bddl_sha256"],
                      "samples": samples, "recorded_success": outcome})
    if seen != {(suite, kind) for suite in SUITES for kind in kinds.values()}:
        raise FixtureError("BenchStudio export requires complete suite coverage")
    return {"schema": "benchstudio.evaluator_qualification.v1",
            "model": "pi0.5" if stack == "pi05" else "smolvla", "runtime": runtime,
            "goal_checker_sha256": checker_sha256, "cases": cases, "passed": True,
            "protocol_e_results_sha256": sha(results)}


def run_fixture_pi05(fixture, root, forbidden_hashes):
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cross_policy"))
    import client_openpi as co  # frozen
    from libero.libero import benchmark, get_libero_path
    suite = benchmark.get_benchmark_dict()[fixture["suite"]]()
    task = suite.get_task(fixture["task_id"])
    state = suite.get_task_init_states(fixture["task_id"])[fixture["initial_state"]["index"]]
    if hashlib.sha256(np.asarray(state).tobytes()).hexdigest() in forbidden_hashes:
        raise EvaluatorIncident("fixture state collides with a confirmation state")
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    text, bddl_sha = bddl_text_and_hash(bddl_path)
    atoms = goal_atoms(text)
    env, _ = co.make_env(task)
    try:
        problem = env.env
        if not same_atoms(atoms, problem.parsed_problem["goal_state"]):
            raise EvaluatorIncident("own BDDL parse differs from the env's goal_state")
        env.reset()
        env.set_init_state(state)
        for _ in range(co.WAIT_STEPS):
            env.step(co.DUMMY_ACTION)
        horizon = co.MAX_STEPS[fixture["suite"]]
        samples = [evaluator_sample(problem, atoms, 0, env.check_success())]
        trace = []
        # The client loop: at most MAX_STEPS steps after settling, stop on done (client_openpi.rollout).
        for step, action in enumerate(actions_for(fixture, horizon, root)[:horizon]):
            _, _, done, _ = env.step(list(action))
            sample = evaluator_sample(problem, atoms, step + 1, done)
            sample["stop_reason"] = (
                ("goal" if sample["goal_met"] else "other") if done
                else "horizon" if step + 1 == horizon else None
            )
            samples.append(sample)
            trace.append({"step": step, "recorded_success": bool(done),
                          "independent_goal": sample["goal_met"], "per_atom": sample["per_atom"],
                          "episode_ended": bool(done)})
            if done:
                break
    finally:
        env.close()
    return trace, horizon, {"bddl_sha256": bddl_sha, "goal_atoms": atoms, "benchstudio_samples": samples}


def run_fixture_smolvla(fixture, root, forbidden_hashes):
    import numpy as np
    sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "cross_policy"), str(Path(__file__).resolve().parents[1])]
    import run_paired as rp  # frozen helpers
    from lerobot.envs import make_env
    from lerobot.envs.configs import LiberoEnv
    from libero.libero import benchmark, get_libero_path
    config = LiberoEnv(task=fixture["suite"], task_ids=[fixture["task_id"]], init_states=True, hard_reset=True,
                       control_mode="relative", observation_height=360, observation_width=360)
    env = make_env(config, n_envs=1, use_async_envs=False)[fixture["suite"]][fixture["task_id"]]
    try:
        from lerobot.envs.utils import NEW_ROLLOUT_OPTION
        base = env.envs[0].unwrapped
        state = np.asarray(base._init_states[fixture["initial_state"]["index"]])
        if hashlib.sha256(state.tobytes()).hexdigest() in forbidden_hashes:
            raise EvaluatorIncident("fixture state collides with a confirmation state")
        # Same substitution as the frozen runner_smolvla.rollout: one-state array, index 0.
        base._init_states = state[None, ...]
        base.init_state_id = 0
        task = benchmark.get_benchmark_dict()[fixture["suite"]]().get_task(fixture["task_id"])
        text, bddl_sha = bddl_text_and_hash(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
        atoms = goal_atoms(text)
        seed = rp.keyed_seed(["protocol_e", fixture["fixture_id"]], domain="environment")
        env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
        problem = base._env.env
        if not same_atoms(atoms, problem.parsed_problem["goal_state"]):
            raise EvaluatorIncident("own BDDL parse differs from the env's goal_state")
        horizon = int(base._max_episode_steps)
        samples = [evaluator_sample(problem, atoms, 0, base._env.check_success())]
        trace = []
        # Step the single underlying env, not the vector wrapper: a vector env may autoreset on termination,
        # and the goal must be read from the state the step produced.
        single = env.envs[0]
        for step, action in enumerate(actions_for(fixture, horizon, root)[:horizon]):
            _, _, terminated, truncated, info = single.step(np.asarray(action, dtype=np.float32))
            recorded = rp.batch_one_success(info)
            ended = bool(terminated) or bool(truncated)
            sample = evaluator_sample(problem, atoms, step + 1, recorded)
            sample["stop_reason"] = (
                "goal" if terminated and sample["goal_met"]
                else "horizon" if step + 1 == horizon and not terminated
                else "other" if ended else None
            )
            samples.append(sample)
            trace.append({"step": step, "recorded_success": recorded,
                          "independent_goal": sample["goal_met"], "per_atom": sample["per_atom"],
                          "episode_ended": ended, "terminated": bool(terminated),
                          "truncated": bool(truncated)})
            if ended:
                break
    finally:
        env.close()
    return trace, horizon, {"bddl_sha256": bddl_sha, "goal_atoms": atoms, "benchstudio_samples": samples}


def main():
    import release
    parser = argparse.ArgumentParser(description="Protocol E evaluator validation (needs Alba's authorisation).")
    parser.add_argument("--stack", required=True, choices=STACKS)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--authorisation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchstudio-runtime", type=Path, help="Separately captured native runtime identities.")
    parser.add_argument("--confirmation-manifest", type=Path, action="append", required=True,
                        help="both models' confirmation state manifests; fixture states must not collide (ISO)")
    args = parser.parse_args()
    if len(args.confirmation_manifest) != 2:
        raise SystemExit("REFUSED: give both models' confirmation manifests for the ISO check")
    # E-AUTH before any env construction (13 E7).
    bound = {"fixtures": args.fixtures, "evaluator_validation.py": __file__}
    if args.benchstudio_runtime is not None:
        bound["benchstudio_runtime"] = args.benchstudio_runtime
    release.check_authorisation(args.authorisation, PROTOCOL_E_DOCUMENT, bound,
                                scope="PROTOCOL_E_" + args.stack.upper())
    root = args.fixtures.resolve().parent
    document = json.loads(args.fixtures.read_text("utf-8"))
    coverage = validate_fixtures(document, args.stack, root)
    forbidden = set()
    for path in args.confirmation_manifest:
        forbidden |= {row["state_sha256"] for row in json.loads(path.read_text("utf-8"))["states"]}
    args.output.mkdir(parents=True, exist_ok=False)
    import os
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    driver = run_fixture_pi05 if args.stack == "pi05" else run_fixture_smolvla
    results, incident = [], None
    for fixture in document["fixtures"]:
        try:
            trace, horizon, meta = driver(fixture, root, forbidden)
        except (EvaluatorIncident, FixtureError, ValueError) as error:
            incident = {"fixture_id": fixture["fixture_id"], "error": repr(error)}
            break
        result = dict(compare(trace, fixture["expected_case"], horizon), fixture_id=fixture["fixture_id"],
                      suite=fixture["suite"], task_id=fixture["task_id"],
                      expected_case=fixture["expected_case"], **meta)
        atomic_write_json(args.output / "{}.trace.json".format(fixture["fixture_id"]), trace)
        results.append(result)
    verdict = stack_verdict(results, coverage["missing_cases"], incident)
    record = {"stack": args.stack, "verdict": verdict, "coverage": coverage, "incident": incident,
              "results": results, "fixtures_sha256": file_sha(args.fixtures), "harness_sha256": file_sha(__file__),
              "results_sha256": sha(results)}
    atomic_write_json(args.output / "protocol_e_record.json", record)
    if args.benchstudio_runtime is not None:
        runtime = json.loads(args.benchstudio_runtime.read_text("utf-8"))
        evidence = benchstudio_report(args.stack, runtime, results, verdict, file_sha(__file__))
        atomic_write_json(args.output / "evaluator.json", evidence)
    print(json.dumps({"stack": args.stack, "verdict": verdict}))


if __name__ == "__main__":
    main()
