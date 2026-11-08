"""Native manifest binding using synthetic states/identities only; no simulator."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import make_schedule_v2 as M
import prepare_campaign as P
import verify_campaign as V
from v3common import Population, file_sha, sha

POP = Population('v3_unit_test', ('synthetic_suite',), (0,), (0,),
                 (990001, 990002), reserved=False)


def fixture(model, state_hash='1' * 64):
    states = [{'key': 'synthetic_suite.task000.slot0', 'suite': 'synthetic_suite',
               'task_id': 0, 'state_slot': 0, 'draw': 1, 'seed': 990003,
               'state_length': 3, 'state_sha256': state_hash,
               'task_bddl_sha256': '2' * 64, 'task_language': 'synthetic task'}]
    manifest = {'schema_version': 1, 'cohort': POP.cohort, 'model': model,
                'state_count': 1, 'states': states, 'states_sha256': sha(states),
                'initial_states_npz_sha256': '3' * 64, 'generator_sha256': '4' * 64,
                'versions': {'synthetic_only': True}}
    return manifest, M.build_schedule(manifest, model, population=POP)


class Binding(unittest.TestCase):
    def test_both_models_accept_matching_native_manifest(self):
        for model in ('pi05', 'smolvla'):
            manifest, schedule = fixture(model)
            M.validate_manifest_binding(schedule, manifest, model, POP)

    def test_rehashed_pair_consistent_rows_still_need_native_binding(self):
        for field, replacement in [('state_sha256', '5' * 64),
                                   ('task_bddl_sha256', '6' * 64),
                                   ('task_language', 'another task'),
                                   ('state_provenance', {'draw': 2, 'seed': 990004, 'state_length': 4})]:
            with self.subTest(field=field):
                manifest, schedule = fixture('pi05')
                for row in schedule['schedule']:
                    row[field] = copy.deepcopy(replacement)
                schedule['schedule_sha256'] = sha(schedule['schedule'])
                M.validate_schedule(schedule['schedule'], POP, 'pi05')
                with self.assertRaisesRegex(M.ManifestError, 'native manifest'):
                    M.validate_manifest_binding(schedule, manifest, 'pi05', POP)

    def test_archive_and_generator_metadata_are_bound(self):
        for field in ('state_manifest_states_sha256', 'initial_states_npz_sha256',
                      'state_generator_sha256'):
            manifest, schedule = fixture('pi05')
            schedule[field] = '9' * 64
            with self.subTest(field=field), self.assertRaises(M.ManifestError):
                M.validate_manifest_binding(schedule, manifest, 'pi05', POP)

    def test_verifier_refuses_before_qualification_or_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'pi05'; root.mkdir()
            archive = root / 'initial_states.npz'; archive.write_bytes(b'synthetic opaque archive')
            manifest, schedule = fixture('pi05')
            manifest['initial_states_npz_sha256'] = file_sha(archive)
            schedule['initial_states_npz_sha256'] = file_sha(archive)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            schedule['state_manifest_file_sha256'] = file_sha(root / 'manifest.json')
            for row in schedule['schedule']:
                row['state_sha256'] = '5' * 64
            schedule['schedule_sha256'] = sha(schedule['schedule'])
            (root / 'schedule.json').write_text(json.dumps(schedule))
            with patch.object(V, 'validate_manifest', side_effect=lambda d, m: M.validate_manifest(d, m, POP)), \
                    patch.object(V, 'pair_table'), \
                    patch.object(V, 'validate_manifest_binding', side_effect=lambda s, d, m: M.validate_manifest_binding(s, d, m, POP)), \
                    patch.object(V.qualification, 'require_qualified') as qualify, \
                    patch.object(V.release, 'check_release') as release:
                with self.assertRaisesRegex(M.ManifestError, 'state_sha256'):
                    V.verify(directory, 'pi05')
                qualify.assert_not_called(); release.assert_not_called()

    def test_preparation_reports_binding_error_without_writing_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'pi05'; root.mkdir()
            manifest, schedule = fixture('pi05')
            (root / 'manifest.json').write_text(json.dumps(manifest))
            schedule['state_manifest_file_sha256'] = file_sha(root / 'manifest.json')
            (root / 'schedule.json').write_text(json.dumps(schedule))
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            with patch('analysis_v2.pair_table'), patch.object(M, 'validate_manifest', return_value={}), \
                    patch.object(M, 'validate_manifest_binding', side_effect=M.ManifestError('native manifest mismatch')):
                report = P.prepare(directory, 'pi05')
            self.assertFalse(report['offline_inputs_complete'])
            self.assertTrue(any(i['message'] == 'Schedule: native manifest mismatch' for i in report['issues']))
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_validator_and_native_verifier_keep_python38_syntax(self):
        for module in (M, V, P):
            ast.parse(Path(module.__file__).read_text(), feature_version=(3, 8))
