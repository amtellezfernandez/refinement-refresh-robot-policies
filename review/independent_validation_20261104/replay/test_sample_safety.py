"""Non-experimental fail-closed tests; all simulator operations are mocked."""
import hashlib
import io
import json
import os
import sys
import tempfile
import tarfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import governance as G
import replay as R
import run_protocol as P


class FakeRep:
    def __init__(self, *args, **kwargs): pass
    def reset(self, *args): return np.array([1.0])
    def close(self): pass


class Safety(unittest.TestCase):
    def test_frozen_protocol_hash(self):
        self.assertEqual(G.sha_file(G.ROOT / 'protocol' / 'PROTOCOL.md'), G.PROTOCOL_SHA256)

    def test_asset_manifest_refuses_changed_or_extra_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); assets = root/'assets'; assets.mkdir()
            (assets/'a').write_text('good')
            manifest = root/'manifest'
            manifest.write_text('./a\t' + G.sha_file(assets/'a') + '\n')
            with patch.object(G, 'ASSET_MANIFEST_SHA256', G.sha_file(manifest)):
                self.assertEqual(G.verify_assets(assets, manifest)['files'], 1)
                (assets/'a').write_text('bad')
                with self.assertRaises(G.ReplayRefused): G.verify_assets(assets, manifest)
                (assets/'a').write_text('good')
                (assets/'unexpected').write_text('extra')
                with self.assertRaises(G.ReplayRefused): G.verify_assets(assets, manifest)

    def test_task_inputs_reject_changed_bddl(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); bddl = root/'task'; bddl.write_text('task')
            init = root/'init'; init.write_text('serialized')
            state = np.array([[1.0]])
            rows = [json.dumps({'episode_identity': {'suite':suite, 'task_id':task},
                                'task_bddl_sha256':G.sha_file(bddl),
                                'initial_state_set_sha256':R.lr.array_sha(state)})
                    for suite in ('libero_goal','libero_object') for task in (5,6,7,8)]
            archive = root/'archive.tar.gz'
            with tarfile.open(archive, 'w:gz') as out:
                for phase in ('confirmation','primary_deadline','confirmation_continuation_partial',
                              'confirmation_continuation_rest'):
                    data = ('\n'.join(rows)+'\n').encode()
                    info = tarfile.TarInfo(phase+'/episodes.jsonl'); info.size = len(data)
                    out.addfile(info, io.BytesIO(data))
            with patch.object(R.lr, 'task_bddl', return_value=(None,None,str(bddl))), \
                 patch.object(R.lr, 'official_init_states', return_value=(state,str(init))):
                self.assertEqual(len(G.verify_task_inputs(archive)),8)
                bddl.write_text('changed')
                with self.assertRaises(G.ReplayRefused): G.verify_task_inputs(archive)

    def test_real_archive_requires_approval(self):
        self.assertIn('staged orchestrator', R.gate([{'provenance':'archived_paper_evidence'}], None, None))

    def test_valid_release_still_cannot_bypass_unresolved_protocol_decision(self):
        with patch.object(G, '_strict_release', return_value=({'scope':[1]}, {}, {}, {})), \
             patch.object(G, 'verify_population', side_effect=AssertionError('must stop before work')):
            with self.assertRaisesRegex(G.ReplayRefused, 'not qualified'):
                G.verify_run([], 'bundles', 'sample', 'release', 'protocol', 1, True,
                             mode='sample_first_57_v1', archive_path='archive')

    def test_sample_mode_missing_approval_refuses_before_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bs = root / 'bundles'; bs.write_text('{"provenance":"archived_paper_evidence"}\n')
            (root/'archive').write_text('synthetic')
            argv = ['run_protocol.py', str(bs), str(root/'sample'), str(root/'out'), '--release',
                    str(root/'missing'), '--protocol', str(root/'protocol'), '--through', '1',
                    '--execution-mode', 'sample_first_57_v1', '--archive', str(root/'archive')]
            with patch.object(sys, 'argv', argv), patch.object(G, 'ARCHIVE_SHA256', G.sha_file(root/'archive')), \
                 self.assertRaises(SystemExit) as cm:
                P.main()
            self.assertEqual(cm.exception.code, 5)
            self.assertFalse((root/'out').exists())

    def test_authorised_mode_selects_exact_manifest_without_320_stage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundles = [{'episode_key': str(i), 'provenance':'archived_paper_evidence',
                        'environment_seed': i} for i in range(320)]
            (root/'bundles').write_text(''.join(json.dumps(b)+'\n' for b in bundles))
            ids = [str(i) for i in range(57)]
            (root/'sample').write_text(json.dumps({'stage1_sample': ids}))
            (root/'archive').write_text('synthetic')
            argv = ['run_protocol.py', str(root/'bundles'), str(root/'sample'), str(root/'out'),
                    '--release', 'mock-release', '--protocol', 'mock-protocol', '--through', '1',
                    '--execution-mode', 'sample_first_57_v1', '--archive', str(root/'archive')]
            calls = []
            def screen(sample, out, render):
                calls.append(('screen', [b['episode_key'] for b in sample]))
                return {'verdict':'PASS_EXACT', 'n':57}
            def episode(stage, sample, *args, **kwargs):
                calls.append((stage, [b['episode_key'] for b in sample]))
                return {'verdict':'PASS_EXACT'}
            with patch.object(sys, 'argv', argv), \
                 patch.object(G, 'ARCHIVE_SHA256', G.sha_file(root/'archive')), \
                 patch.object(G, 'verify_run', return_value={'release_sha256':'mock'}), \
                 patch.object(P, 'sample_reset_screen', side_effect=screen), \
                 patch.object(P, 'episode_stage', side_effect=episode), \
                 patch.object(P, 'stage0', side_effect=AssertionError('full stage called')):
                with self.assertRaises(SystemExit) as cm: P.main()
            self.assertEqual(cm.exception.code, 0)
            self.assertEqual(calls, [('screen', ids), (1, ids)])

    def test_reset_failure_stops_next_reset_and_seals_rows(self):
        bs = [{'episode_key': str(i), 'suite': 's', 'task_id': 1, 'environment_seed': i,
               'recorded_initial_state': {'values': [2.0], 'dtype': '<f8', 'shape': [1]}} for i in range(3)]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.object(R, 'validate', return_value=[]), patch.object(P.lr, 'Replayer', FakeRep), \
                 patch.object(R, 'pre_settle_state', return_value=(np.array([1.0]), {})):
                result = P.sample_reset_screen(bs, root, True)
            self.assertEqual(result, {'verdict':'FAIL_SAMPLE_RESET', 'n':1})
            self.assertEqual(len((root / 'sample_reset_screen.jsonl').read_text().splitlines()), 1)
            self.assertIn(hashlib.sha256((root / 'sample_reset_screen.jsonl').read_bytes()).hexdigest(),
                          (root / 'SHA256SUMS').read_text())

    def test_interrupted_screen_retains_checked_rows_and_hashes(self):
        bs = [{'episode_key': str(i), 'suite': 's', 'task_id': 1, 'environment_seed': i,
               'recorded_initial_state': {'values': [1.0], 'dtype': '<f8', 'shape': [1]}} for i in range(3)]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.object(R, 'validate', side_effect=[[], KeyboardInterrupt]), \
                 patch.object(P.lr, 'Replayer', FakeRep), \
                 patch.object(R, 'pre_settle_state', return_value=(np.array([1.0]), {})):
                with self.assertRaises(KeyboardInterrupt): P.sample_reset_screen(bs, root, True)
            data = (root / 'sample_reset_screen.jsonl').read_bytes()
            self.assertEqual(len(data.splitlines()), 2)
            self.assertEqual(json.loads(data.splitlines()[-1])['verdict'], 'INTERRUPTED')
            self.assertIn(hashlib.sha256(data).hexdigest(), (root / 'SHA256SUMS').read_text())

    def test_failed_attempt_stops_without_retries_or_skips(self):
        bs = [{'episode_key':str(i), 'provenance':'archived_paper_evidence', 'actions':[]} for i in range(3)]
        calls = []
        def validate(b):
            calls.append(b['episode_key'])
            return ['defect'] if b['episode_key'] == '1' else []
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def replay(rep, b, mode, cache):
                return {'reset_state_max_abs_diff':0, 'replay_success':False, 'recorded_success':False,
                        'first_divergent_tick':None, 'reset_state_exact':True}
            for b in bs: b.update(suite='s', task_id=1)
            with patch.object(R, 'gate', return_value=None), patch.object(R, 'validate', side_effect=validate), \
                 patch.object(R.lr, 'Replayer', FakeRep), patch.object(R, 'replay_one', side_effect=replay):
                code = R.run(bs, root / 'arm', artifact_paths={'evidence_root':str(root)})
            rows = [json.loads(x) for x in (root / 'arm/results.jsonl').read_text().splitlines()]
            self.assertEqual(code, 6)
            self.assertEqual(calls, ['0', '1'])
            self.assertEqual([x['episode_key'] for x in rows], ['0', '1'])
            self.assertEqual(rows[-1]['verdict'], 'INPUT_INVALID')
            self.assertIn(hashlib.sha256((root / 'arm/results.jsonl').read_bytes()).hexdigest(),
                          (root / 'SHA256SUMS').read_text())

    def test_interrupted_attempt_is_logged_and_checksummed(self):
        b = {'episode_key':'one', 'provenance':'archived_paper_evidence', 'actions':[],
             'suite':'s', 'task_id':1}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.object(R, 'gate', return_value=None), patch.object(R, 'validate', return_value=[]), \
                 patch.object(R.lr, 'Replayer', FakeRep), patch.object(R, 'replay_one', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    R.run([b], root/'arm', artifact_paths={'evidence_root':str(root)})
            data = (root/'arm/results.jsonl').read_bytes()
            self.assertEqual(json.loads(data)['verdict'], 'UNREPLAYABLE')
            self.assertIn('KeyboardInterrupt', json.loads(data)['error'])
            self.assertIn(hashlib.sha256(data).hexdigest(), (root/'SHA256SUMS').read_text())

    def test_sample_outcome_mismatch_stops_before_next_attempt(self):
        bs = [{'episode_key':str(i), 'provenance':'archived_paper_evidence', 'actions':[],
               'suite':'s', 'task_id':1} for i in range(3)]
        calls = []
        def mismatch(rep, b, mode, cache):
            calls.append(b['episode_key'])
            return {'reset_state_max_abs_diff':0, 'replay_success':False, 'recorded_success':True,
                    'first_divergent_tick':None, 'reset_state_exact':True}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.object(R, 'gate', return_value=None), patch.object(R, 'validate', return_value=[]), \
                 patch.object(R.lr, 'Replayer', FakeRep), patch.object(R, 'replay_one', side_effect=mismatch):
                code = R.run(bs, root/'arm', artifact_paths={'evidence_root':str(root), 'mode':'sample_first_57_v1'})
            self.assertEqual(code, 6)
            self.assertEqual(calls, ['0'])
            self.assertEqual(json.loads((root/'arm/results.jsonl').read_text())['verdict'], 'OUTCOME_MISMATCH')

    def test_nondeterminism_stops_during_second_arm(self):
        bs = [{'episode_key':str(i), 'provenance':'archived_paper_evidence', 'actions':[],
               'suite':'s', 'task_id':1} for i in range(2)]
        calls = []
        def changed(rep, b, mode, cache):
            calls.append(b['episode_key'])
            return {'reset_state_max_abs_diff':0, 'replay_success':False, 'recorded_success':False,
                    'first_divergent_tick':None, 'reset_state_exact':True, 'terminal_state_sha256':'changed'}
        reference = [{'episode_key':b['episode_key'], 'replay':{'terminal_state_sha256':'original'}} for b in bs]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.object(R, 'gate', return_value=None), patch.object(R, 'validate', return_value=[]), \
                 patch.object(R.lr, 'Replayer', FakeRep), patch.object(R, 'replay_one', side_effect=changed):
                code = R.run(bs, root/'arm', artifact_paths={'evidence_root':str(root), 'mode':'sample_first_57_v1'},
                             reference_records=reference)
            self.assertEqual(code, 6)
            self.assertEqual(calls, ['0'])
            self.assertEqual(json.loads((root/'arm/results.jsonl').read_text())['verdict'], 'NONDETERMINISTIC')

if __name__ == '__main__': unittest.main()
