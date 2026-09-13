"""Offline design/selection tests; synthetic outcomes never enter the campaign."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from last_campaign import CONF, DEV, GRID, SEEDS, UNTESTED, plan, selection
from run_paired import select_schedule, sha


class CampaignTests(unittest.TestCase):
    def test_pairing_and_reservations(self):
        discovery = plan(DEV, range(10,15), SEEDS, GRID, "discovery")
        self.assertEqual(len(select_schedule(discovery)),480)
        self.assertEqual(len(select_schedule(discovery,8)),96)
        self.assertFalse(set(DEV)&set(CONF))
        self.assertFalse(set(GRID)&set(UNTESTED))
        broken=copy.deepcopy(discovery)
        broken["schedule"].pop()
        broken["scheduling"]["schedule_sha256"]=sha(broken["schedule"])
        with self.assertRaises(ValueError): select_schedule(broken)

    def test_selection_freezes_before_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"discovery").mkdir()
            rows=plan(DEV,range(10,15),SEEDS,GRID,"synthetic")["schedule"]
            for row in rows:
                row["success"]=(row["h"]==1 and row["denoising_steps"]>=5)
            (root/"discovery/episodes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            costs={str(d):100+10*d for d in (2,3,5,8,10,20)}
            frozen=selection(SimpleNamespace(output=root),costs)
            self.assertTrue((root/"frozen_prediction.json").is_file())
            self.assertFalse((root/"confirmation").exists())
            for comparison in frozen["comparisons"]:
                if "recommendation" not in comparison: continue
                for field in ("recommendation","alternative"):
                    h,d=comparison[field]
                    self.assertLessEqual(costs[str(d)]/h,comparison["ceiling"])
            confirmation=json.loads((root/"confirmation_plan.json").read_text())
            self.assertLessEqual(len(select_schedule(confirmation)),1120)
            self.assertEqual(len(select_schedule(confirmation))%160,0)


if __name__=="__main__": unittest.main()
