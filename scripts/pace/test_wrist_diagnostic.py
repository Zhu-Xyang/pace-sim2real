"""Offline checks for the optional wrist-hold collector (no simulator launch)."""

import csv
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

import wrist_diagnostic as wd
from experiment_utils import group_chirp, select_groups
from s800_motion import nominal_motion


class WristDiagnosticTests(unittest.TestCase):
    def test_asset_changes_only_four_joint_types(self):
        before = ET.parse(wd.URDF_DIR / "serial_robot_s_pace_22dof.urdf").getroot()
        after = ET.parse(wd.URDF).getroot()
        for name in wd.WRISTS:
            before.find(f"joint[@name='{name}']").set("type", "revolute")
        self.assertEqual(ET.tostring(before), ET.tostring(after))
        self.assertEqual(sum(j.get("type") != "fixed" for j in after.findall("joint")), 26)

    def test_config_mapping_and_held_wrist_targets(self):
        root = ET.parse(wd.URDF_DIR / "serial_robot_s_pace_22dof.urdf").getroot()
        old = [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]
        bounds = torch.arange(178).reshape(89, 2).float()
        base = NS(joint_names_expr=[".*"])
        cfg = NS(sim2real=NS(joint_order=old.copy(), bounds_params=bounds.clone()),
                 scene=NS(robot=NS(spawn=NS(usd_path="old"), actuators={"joints": base})))
        with tempfile.TemporaryDirectory() as td:
            usd = Path(td) / "test.usda"
            usd.touch()
            with patch.object(wd, "USD", usd):
                info = wd.configure(cfg, 100., 7.)
        names = cfg.sim2real.joint_order
        self.assertEqual(base.joint_names_expr, [".*"])
        for block in range(4):
            for i, name in enumerate(old):
                torch.testing.assert_close(cfg.sim2real.bounds_params[block*26+names.index(name)], bounds[block*22+i])
        torch.testing.assert_close(cfg.sim2real.bounds_params[-1], bounds[-1])
        self.assertEqual(cfg.scene.robot.actuators["wrists"].saturation_effort, 5.)
        self.assertEqual(len(info["urdf_effort_limit"]), 26)
        _, selected = select_groups("arms", names)
        wrists = [names.index(n) for n in wd.WRISTS]
        selected = [i for i in selected if i not in wrists]
        self.assertEqual(len(selected), 10)
        center, amplitude = nominal_motion(names)
        center[wrists] = amplitude[wrists] = 0
        t, target = group_chirp(.005, 2., .1, .2, .1, torch.full((1, 26), 2.), center, amplitude, selected)
        self.assertEqual(int(torch.count_nonzero(target[:, wrists])), 0)
        self.assertGreater(float(target[:, selected].std(dim=0).min()), .001)

    def test_raw_request_exceeds_limit_even_when_applied_is_clipped(self):
        target = torch.tensor([[.06], [0.]])
        q = torch.zeros_like(target)
        v = torch.tensor([[0.], [-1.]])
        request = wd.predicted_pd(target, q, v, torch.tensor([100.]), torch.tensor([7.]))
        torch.testing.assert_close(request, torch.tensor([[6.], [7.]]))
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "sample.pt"
            rows = wd.save_torque_report(out, torch.tensor([0., .005]), [wd.WRISTS[0]], request,
                                         request, request.clamp(-5, 5), torch.tensor([100.]), torch.tensor([7.]),
                                         torch.tensor([5.]), torch.tensor([5.]))
            self.assertEqual(rows[0]["reference_exceed_percent"], 100.)
            self.assertEqual(rows[0]["applied_peak_nm"], 5.)
            self.assertEqual(rows[0]["first_reference_exceed_s"], 0.)
            with out.with_suffix(".torque.csv").open() as f:
                self.assertEqual(len(list(csv.reader(f))), 3)


if __name__ == "__main__":
    unittest.main()
