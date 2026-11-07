"""Single operator command for a separately released R22/R25 campaign.

Frontend runs in BenchStudio's Python >=3.10 environment. Native commands run in
independent pinned interpreters. No qualification or release is created here.
"""
import argparse
from pathlib import Path
import shlex
import socket
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'experiments/tooling/cross_policy_v3'
XLA_FLAGS = '--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'


def canonical_workspace(root=ROOT):
    # Existing launch_guard.verify_released_schedule requires this directory.
    return Path(root).resolve() / 'experiments/runs/cross_policy_confirmation_v2'


def plan(workspace, smolvla_python, pi05_python, *, port=8000, root=ROOT):
    workspace, root = Path(workspace).resolve(), Path(root).resolve()
    if workspace != canonical_workspace(root).resolve():
        raise ValueError('Use the existing released schedule directory: ' + str(canonical_workspace(root)))
    tools = root / 'experiments/tooling/cross_policy_v3'
    commands = []
    interpreters = {'smolvla': str(smolvla_python), 'pi05': str(pi05_python)}
    for model, python in interpreters.items():
        commands.append((model + '-verify', [python, str(tools / 'verify_campaign.py'),
                                           '--workspace', str(workspace), '--model', model]))
    for model, python in interpreters.items():
        folder = workspace / model
        if model == 'pi05':
            commands.append(('pi05-server', [python, str(tools / 'server_openpi_v3.py'), 'serve',
                '--checkpoint', str(folder / 'checkpoint'), '--output', str(folder / 'server'),
                '--schedule', str(folder / 'schedule.json'), '--port', str(port)]))
        command = [python, str(tools / 'launch_v3.py'), '--model', model,
            '--schedule', str(folder / 'schedule.json'), '--states', str(folder / 'initial_states.npz'),
            '--output', str(folder / 'run')]
        command += (['--checkpoint', str(folder / 'checkpoint')] if model == 'smolvla' else
                    ['--startup-gate', str(folder / 'server/startup_gate.json'), '--port', str(port)])
        commands.append((model + '-campaign', command))
        command = [python, str(tools / 'analysis_v2.py'), '--schedule', str(folder / 'schedule.json'),
            '--summary', str(folder / 'run/run_summary.json'), '--journal', str(folder / 'run/attempts'),
            '--output', str(folder / 'analysis.json')]
        if (folder / 'r3_record.json').is_file():
            command += ['--r3-record', str(folder / 'r3_record.json')]
        commands.append((model + '-analysis', command))
    return commands


def execute(commands, workspace, *, port=8000, startup_timeout=1800, root=ROOT):
    import benchstudio as bs
    workspace = Path(workspace).resolve()
    # Refuse a second launch, rather than overwrite scientific outputs/logs.
    # This is an output-preservation check, not a global single-campaign registry.
    for model in ('smolvla', 'pi05'):
        folder = workspace / model
        if not (folder / 'checkpoint').is_dir():
            raise ValueError('Missing pinned checkpoint directory: ' + str(folder / 'checkpoint'))
        if any((folder / path).exists() for path in ('run', 'analysis.json', 'server')):
            raise ValueError('Existing output: use the reviewed recovery path; do not rerun or move it')
    by_name = dict(commands)
    log_root = workspace / 'operator-logs'
    log_root.mkdir(parents=True, exist_ok=True)
    logs = Path(tempfile.mkdtemp(prefix='launch-', dir=log_root))
    print('Operator logs: ' + str(logs), flush=True)
    env = {'MUJOCO_GL': 'osmesa', 'XLA_FLAGS': XLA_FLAGS}
    def command(name):
        return bs.ProcessCommand(name, tuple(by_name[name]), root, env=env)
    # Both native stacks must pass before the first model/server is started.
    for model in ('smolvla', 'pi05'):
        command(model + '-verify').run(logs)
    # Reject an occupied service port before spending a model run.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))
    def model_round(model, required_service=None):
        try:
            command(model + '-campaign').run(logs, required_service=required_service)
        except bs.ProcessFailure:
            # A stopped native run may have an accounting-only summary. Its
            # existing analysis decides whether any outcomes can be read.
            if (workspace / model / 'run/run_summary.json').is_file():
                command(model + '-analysis').run(logs)
            raise
        command(model + '-analysis').run(logs)
    model_round('smolvla')
    gate = workspace / 'pi05/server/startup_gate.json'
    def ready():
        if not gate.is_file():
            return False
        # The server binds its port only after its existing compiled-count gate.
        with socket.socket() as probe:
            probe.settimeout(0.2)
            return probe.connect_ex(('127.0.0.1', port)) == 0
    with command('pi05-server').service(logs, ready=ready, startup_timeout=startup_timeout) as server:
        model_round('pi05', required_service=server)
    print('Both model runs and their separate offline analyses completed. See ' + str(workspace))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=canonical_workspace(),
                        help='Existing release-gate directory; normally omit this argument.')
    parser.add_argument('--smolvla-python', required=True)
    parser.add_argument('--pi05-python', required=True)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--startup-timeout', type=float, default=1800)
    parser.add_argument('--execute', action='store_true', help='Invoke existing release-gated native entry points.')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.startup_timeout <= 0:
        parser.error('port/timeout outside valid range')
    commands = plan(args.workspace, args.smolvla_python, args.pi05_python, port=args.port)
    if not args.execute:
        for name, argv in commands:
            print(name + ': ' + shlex.join(argv))
        return
    execute(commands, args.workspace, port=args.port, startup_timeout=args.startup_timeout)


if __name__ == '__main__':
    main()
