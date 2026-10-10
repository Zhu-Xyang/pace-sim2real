"""CPU regressions for segmented identification. Run with python -m unittest discover -s scripts/pace -p 'test_*.py'."""

import ast
import importlib.util
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
import numpy as np

from experiment_utils import (build_joint_groups, select_groups, group_chirp,
                              freeze_unfitted_truth, validate_recording, recording_name)

ROOT = Path(__file__).resolve().parents[2]


class ExperimentTests(unittest.TestCase):
    def test_plot_table_uses_snapshot_truth_instead_of_manual_constants(self):
        import csv
        path = ROOT / "scripts/pace/plot_trajectory.py"
        tree = ast.parse(path.read_text())
        block = next(n for n in tree.body if isinstance(n, ast.If)
                     and isinstance(n.test, ast.Name) and n.test.id == "plot_table")
        names = ["left", "right"]
        gt = dict(joint_order=names, armature=torch.tensor([.1, .2]),
                  damping=torch.tensor([.3, .4]), friction=torch.tensor([.5, .6]),
                  bias=torch.tensor([.01, .02]), delay=3.)
        with tempfile.TemporaryDirectory() as directory:
            values = dict(plot_table=True, joint_order=names, config={"gt": gt},
                          mean=torch.ones(9), log_dir=Path(directory), csv=csv,
                          GT_ARMATURE={}, GT_VISCOUS={}, GT_FRICTION={}, GT_BIAS=.05, GT_DELAY=5)
            with patch("builtins.print"):
                exec(compile(ast.Module(body=[block], type_ignores=[]), str(path), "exec"), values)
            with (Path(directory) / "param_comparison.csv").open() as file:
                rows = list(csv.reader(file))
            self.assertAlmostEqual(float(rows[1][1]), .1)
            self.assertAlmostEqual(float(rows[1][10]), .01)
            self.assertAlmostEqual(float(rows[2][10]), .02)
            self.assertEqual(float(rows[-1][1]), 3.)

    def test_scaled_legacy_preserves_original_formula_and_unexcited_joints(self):
        from s800_motion import nominal_motion, scaled_nominal_motion
        names = ["J01_HIP_ROLL_L", "J07_HIP_ROLL_R", "J03_KNEE_PITCH_L", "J12_TORSO_YAW"]
        c, a = nominal_motion(names)
        cc, aa = scaled_nominal_motion(names, [0, 1, 2], .8)
        for wave in (-1., 0., 1.):
            torch.testing.assert_close((cc+wave*aa)[:3], ((c+wave*a)*.8)[:3])
        self.assertAlmostEqual(float(cc[0]), .036)
        self.assertAlmostEqual(float(cc[1]), -.036)
        self.assertAlmostEqual(float(cc[2]), .4184)
        torch.testing.assert_close(cc[3:], c[3:])
        torch.testing.assert_close(aa[3:], a[3:])
        for invalid in (0., -1., 1.1, float("nan")):
            with self.assertRaises(ValueError):
                scaled_nominal_motion(names, [0], invalid)

    def test_box_separation_including_rotated_and_parallel_axes(self):
        from s800_clearance import box_gap, rotation_rpy
        eye = np.eye(3)[None]
        half = np.ones(3)
        for x, expected in ((3., 1.), (2., 0.), (1., -1.)):
            gap = box_gap(np.zeros((1, 3)), eye, half, np.array([[x, 0., 0.]]), eye, half)
            self.assertAlmostEqual(float(gap[0]), expected)
        rotated = rotation_rpy([0., 0., np.pi/4])[None]
        gap = box_gap(np.zeros((1, 3)), eye, half, np.array([[4., 0., 0.]]), rotated, half)
        self.assertAlmostEqual(float(gap[0]), 3.-np.sqrt(2))

    def test_leg_plan_preserves_center_and_rejects_crossed_legacy_sweep(self):
        from s800_clearance import LegClearance, with_midpoints
        from s800_motion import nominal_motion, choose_leg_motion
        path = ROOT / "source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/s800_pace_env_cfg.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
                    and isinstance(n.target, ast.Name) and n.target.id == "joint_order")
        names = ast.literal_eval(node.value)
        checker = LegClearance(ROOT / "assets_robot/engineai_S800/urdf/serial_s800.urdf", names)
        center, amplitude = nominal_motion(names)
        self.assertAlmostEqual(float(center[1]), .045)
        self.assertAlmostEqual(float(center[3]), .523)
        wave = np.zeros((401, len(names)))
        wave[:, :12] = np.linspace(-1., 1., 401)[:, None]
        bias = np.full(len(names), .05)
        self.assertFalse(checker.check(center.numpy()+wave*amplitude.numpy()+bias)["passed"])
        c, a, report = choose_leg_motion(checker, wave, center, amplitude, bias, list(range(12)), .02)
        self.assertTrue(report["passed"])
        self.assertTrue(checker.check(with_midpoints(c.numpy()+wave*a.numpy()+bias), margin=.05)["passed"])
        torch.testing.assert_close(c[[0, 3, 6, 9]], center[[0, 3, 6, 9]])
        torch.testing.assert_close(c[12:], center[12:])
        torch.testing.assert_close(a[12:], amplitude[12:])
        self.assertAlmostEqual(float(a[0]), .3)
        self.assertAlmostEqual(float(a[5]), .0315)
        # A midpoint is an extra sampled check, not a continuous-motion certificate.
        np.testing.assert_allclose(with_midpoints(np.array([[0., 2.], [2., 4.]])),
                                   [[0., 2.], [1., 3.], [2., 4.]])

    def test_command_delay_uses_current_feedback_and_torque_delay_does_not(self):
        # Isolate placement of delay from the simulator and from the DC saturation model.
        class Buffer:
            def __init__(self, *args, **kwargs):
                self.history = []

            def set_time_lag(self, delay, env_ids=None):
                self.delay = int(delay)

            def reset(self, env_ids=None):
                self.history.clear()

            def compute(self, value):
                self.history.append(value.clone())
                return self.history[max(0, len(self.history) - 1 - self.delay)].clone()

        class Motor:
            def __init__(self, cfg):
                self._num_envs, self.num_joints, self._device = 1, 1, "cpu"

            def _parse_joint_parameter(self, value, default):
                return torch.tensor([[value]])

            def reset(self, env_ids):
                pass

            def compute(self, action, pos, vel):
                self.applied_effort = 2 * (action.joint_positions - pos) - vel
                action.joint_efforts = self.applied_effort
                return action

        modules = {name: types.ModuleType(name) for name in
                   ("isaaclab", "isaaclab.actuators", "isaaclab.utils", "isaaclab.utils.types")}
        modules["isaaclab.actuators"].DCMotor = Motor
        modules["isaaclab.utils"].DelayBuffer = Buffer
        modules["isaaclab.utils.types"].ArticulationActions = types.SimpleNamespace
        path = ROOT / "source/pace_sim2real/pace_sim2real/utils/pace_actuator.py"
        spec = importlib.util.spec_from_file_location("pace_actuator_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", modules):
            spec.loader.exec_module(module)
        for lag in (0, 1):
            outputs = []
            for cls in (module.PaceDCMotor, module.PaceDCMotorCmdDelay):
                actuator = cls(types.SimpleNamespace(encoder_bias=0., max_delay=lag))
                actuator.reset([0])
                for target, pos in ((1., 0.), (4., .5)):
                    action = types.SimpleNamespace(joint_positions=torch.tensor([[target]]),
                                                   joint_velocities=torch.zeros(1, 1), joint_efforts=torch.zeros(1, 1))
                    result = actuator.compute(action, torch.tensor([[pos]]), torch.zeros(1, 1))
                self.assertEqual(float(actuator.applied_effort), float(result.joint_efforts))
                outputs.append(float(result.joint_efforts))
            self.assertEqual(outputs, [7., 7.] if lag == 0 else [2., 1.])

    def test_actual_s800_partition(self):
        path = ROOT / "source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/s800_pace_env_cfg.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
                    and isinstance(n.target, ast.Name) and n.target.id == "joint_order")
        names = ast.literal_eval(node.value)
        groups = build_joint_groups(names)
        self.assertEqual([len(groups[g]) for g in ("legs", "torso", "arms")], [12, 1, 14])
        self.assertEqual(sorted(j for values in groups.values() for j in values), list(range(27)))
        with self.assertRaises(ValueError):
            select_groups("all,legs", names)
        self.assertNotEqual(recording_name("legs", "command"), recording_name("arms", "command"))

    def test_chirp_holds_other_joints_and_has_exact_timing(self):
        center = torch.tensor([0.1, -0.2, 0.3])
        time, commands = group_chirp(.0025, 1., .1, .1, .1, torch.tensor([2., 2., 2.]),
                                     center, torch.ones(3), [1])
        self.assertEqual(len(time), 480)
        torch.testing.assert_close(time, torch.arange(480) * .0025)
        torch.testing.assert_close(commands[:, [0, 2]], center[[0, 2]].repeat(480, 1))
        torch.testing.assert_close(commands[:40], center.repeat(40, 1))
        torch.testing.assert_close(commands[-40:], center.repeat(40, 1))
        self.assertGreater(float(commands[:, 1].std()), .1)
        torch.testing.assert_close(commands[40], center)
        torch.testing.assert_close(commands[-41], center)

    def test_frozen_truth_does_not_initialize_fitted_parameters(self):
        names = ["HIP_PITCH", "TORSO_YAW", "SHOULDER_PITCH"]
        data = {"gt": {"joint_order": names, "delay": 3.,
                       **{key: torch.tensor([.2, .3, .4]) for key in ("armature", "damping", "friction", "bias")}}}
        bounds = torch.tensor([[0., 1.]] * 12 + [[0., 10.]])
        result = freeze_unfitted_truth(bounds, data, names, [1])
        torch.testing.assert_close(result[[1, 4, 7, 10]], torch.full((4,), .5))
        self.assertEqual(result[-1], 5.)
        torch.testing.assert_close(result[[0, 2]], torch.tensor([.2, .4]))

    def test_replay_rejects_wrong_timing_gain_mode_and_excitation(self):
        names = ["HIP_PITCH", "TORSO_YAW"]
        current = dict(physics_dt=.0025, decimation=1, delay_mode="command", kp=torch.ones(2))
        data = dict(time=torch.arange(4) * .0025, joint_names=names, dof_pos=torch.zeros(4, 2),
                    des_dof_pos=torch.zeros(4, 2), excited=names[:1], experiment=current.copy())
        validate_recording(data, names, [0], current)
        for key, value in (("physics_dt", .002), ("delay_mode", "torque"), ("kp", torch.zeros(2))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_recording(data, names, [0], {**current, key: value})
        with self.assertRaises(ValueError):
            validate_recording(data, names, [1], current)
        data["experiment"]["actual_clearance"] = {"passed": False}
        with self.assertRaisesRegex(ValueError, "clearance"):
            validate_recording(data, names, [0], current)

    def test_optimizer_loss_mask_freezing_and_checkpoint_pair(self):
        path = ROOT / "source/pace_sim2real/pace_sim2real/optim/cma_es.py"
        spec = importlib.util.spec_from_file_location("pace_optimizer_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            bounds = torch.tensor([[0., 1.]] * 12 + [[0., 10.]])
            data = dict(dof_pos=torch.zeros(2, 3), des_dof_pos=torch.zeros(2, 3), time=torch.arange(2))
            opt = module.CMAESOptimizer(bounds, 4, directory, ["leg", "torso", "arm"], 1, data,
                                       "cpu", groups={"legs": [0], "torso": [1], "arms": [2]},
                                       active_groups=["torso"], loss_joints=[1], opt_delay=False,
                                       segment_edges_hz=(), save_interval=1)
            self.assertEqual(opt.active_idx.tolist(), [1, 4, 7, 10])
            torch.testing.assert_close(opt.sim_params[:, 0], torch.full((4,), .5))
            residual = torch.tensor([[1000., 1., 1000.], [1000., 2., 1000.],
                                     [1000., 3., 1000.], [1000., 4., 1000.]])
            expected_best = opt.sim_params[0].clone()
            for _ in range(2):
                opt.tell(opt.sim_params[:, opt.bias_idx] + residual, torch.zeros(4, 3))
            torch.testing.assert_close(opt.scores, torch.tensor([2., 8., 18., 32.]))
            expected_traj = opt.sim_dof_pos_buffer[0].clone()
            opt.evolve()
            self.assertTrue(opt.finished())
            run = Path(opt.writer.log_dir)
            torch.testing.assert_close(torch.load(run / "best_params.pt", weights_only=True), expected_best)
            torch.testing.assert_close(torch.load(run / "best_trajectory.pt", weights_only=True), expected_traj)
            opt.close()


if __name__ == "__main__":
    unittest.main()
