"""Synthetic-only regression tests for PR #3 stage accounting and stops."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_protocol as P
import replay as R


def row(key):
    return {'episode_key':key,'verdict':'REPRODUCED','replay':{
        'terminal_state_sha256':'synthetic','replay_success':False,'recorded_success':False,
        'replay_first_success_tick':None,'recorded_first_success_tick':None,'first_divergent_tick':None}}

class StageAccounting(unittest.TestCase):
    def test_missing_duplicate_unexpected_reordered_and_invalid_rows_refused(self):
        bundles=[{'episode_key':str(i),'condition':'h2_d2','label':{'success':False}} for i in range(2)]
        contradictory=row('1'); contradictory['replay']['first_divergent_tick']=3
        hidden_defect=row('1'); hidden_defect['input_defects']=['synthetic defect']
        wrong_label=row('1'); wrong_label['replay'].update(recorded_success=True, recorded_first_success_tick=3)
        for name, rows in [('missing',[row('0')]),('duplicate',[row('0'),row('0')]),
                           ('unexpected',[row('0'),row('X')]),('reordered',[row('1'),row('0')]),
                           ('invalid',[row('0'),{'episode_key':'1','verdict':'REPRODUCED','replay':{}}]),
                           ('contradictory',[row('0'),contradictory]),
                           ('hidden_defect',[row('0'),hidden_defect]),
                           ('wrong_label',[row('0'),wrong_label])]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                def fake(bs,out,**kwargs):
                    path=Path(out); path.mkdir(parents=True); (path/'results.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows)); return 0
                with patch.object(R,'run',side_effect=fake):
                    result=P.episode_stage(1,bundles,Path(td),False,None,None,'PASS_EXACT')
                self.assertEqual(result['verdict'],'IMPLEMENTATION_FAILURE')

    def test_duplicate_scheduled_identity_stops_before_first_arm(self):
        bundles=[{'episode_key':'0'},{'episode_key':'0'}]
        with tempfile.TemporaryDirectory() as td, patch.object(R,'run',side_effect=AssertionError('must not run')):
            result=P.episode_stage(1,bundles,Path(td),False,None,None,'PASS_EXACT')
        self.assertEqual(result['verdict'],'IMPLEMENTATION_FAILURE')

    def test_early_failure_types_stay_distinct(self):
        bundle=[{'episode_key':'0'}]
        for raw, expected in [('INPUT_INVALID','ARCHIVE_DEFECT'),('DUPLICATE','ARCHIVE_DEFECT'),
                              ('OUTCOME_MISMATCH','STOPPED'),('UNREPLAYABLE','UNINTERPRETABLE'),
                              ('REFERENCE_MISMATCH','IMPLEMENTATION_FAILURE')]:
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as td:
                def fake(bs,out,**kwargs):
                    path=Path(out); path.mkdir(parents=True); (path/'results.jsonl').write_text(json.dumps({'episode_key':'0','verdict':raw})+'\n'); return 6
                with patch.object(R,'run',side_effect=fake):
                    result=P.episode_stage(1,bundle,Path(td),False,None,None,'PASS_EXACT')
                self.assertEqual(result['verdict'],expected)
                self.assertEqual('note' in result, raw=='OUTCOME_MISMATCH')

    def test_interrupt_is_recorded_and_reraised(self):
        with tempfile.TemporaryDirectory() as td, patch.object(R,'run',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                P.episode_stage(1,[{'episode_key':'0'}],Path(td),False,None,None,'PASS_EXACT')
            self.assertEqual(json.loads((Path(td)/'stage1/stage_verdict.json').read_text())['verdict'],'INTERRUPTED')

    def test_approval_refusal_remains_distinct(self):
        with tempfile.TemporaryDirectory() as td, patch.object(R,'run',return_value=5):
            result=P.episode_stage(1,[{'episode_key':'0'}],Path(td),False,None,None,'PASS_EXACT')
        self.assertEqual(result['verdict'],'REFUSED')

    def test_implementation_exception_is_not_experimental_stop(self):
        with tempfile.TemporaryDirectory() as td, patch.object(R,'run',side_effect=RuntimeError('injected')):
            result=P.episode_stage(1,[{'episode_key':'0'}],Path(td),False,None,None,'PASS_EXACT')
        self.assertEqual(result['verdict'],'IMPLEMENTATION_FAILURE')

if __name__=='__main__': unittest.main()
