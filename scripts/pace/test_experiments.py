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
                              freeze_unfitted_truth, validate_recording, recording_name,
                              expand_fit_bounds, select_leg_sweep)

ROOT = Path(__file__).resolve().parents[2]


class ExperimentTests(unittest.TestCase):
    def test_split_leg_optimizer_only_changes_six_joints(self):
        import json
        names = json.loads((ROOT / 'scripts/pace/trajectory_plans/s800_legs_sim_20261012_22dof.json').read_text())['joint_order']
        groups = build_joint_groups(names)
        legs = groups.pop('legs')
        groups.update({g: select_leg_sweep(names, g) for g in ('hips', 'knee_ankle')})
        self.assertEqual(groups['hips'], [0, 1, 2, 6, 7, 8])
        self.assertEqual(groups['knee_ankle'], [3, 4, 5, 9, 10, 11])
        self.assertEqual(sorted(groups['hips'] + groups['knee_ankle']), legs)
        spec = importlib.util.spec_from_file_location('split_optimizer_test', ROOT / 'source/pace_sim2real/pace_sim2real/optim/cma_es.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from experiment_utils import LEG_JOINT_TYPES
        groups.update({g: select_leg_sweep(names, g) for g in LEG_JOINT_TYPES})
        for group in ('hips', 'knee_ankle', *LEG_JOINT_TYPES):
            with tempfile.TemporaryDirectory() as directory:
                bounds = torch.tensor([[0., 1.]] * 88 + [[0., 10.]])
                data = dict(dof_pos=torch.zeros(2, 22), des_dof_pos=torch.zeros(2, 22), time=torch.arange(2))
                opt = module.CMAESOptimizer(bounds, 4, directory, names, 1, data, 'cpu',
                                           groups=groups, active_groups=[group], loss_joints=groups[group],
                                           opt_delay=True, segment_edges_hz=())
                expected = sorted([b*22+i for b in range(4) for i in groups[group]] + [88])
                self.assertEqual(opt.active_idx.tolist(), expected)
                frozen = [i for i in range(89) if i not in expected]
                torch.testing.assert_close(opt.sim_params[:, frozen], torch.full((4, len(frozen)), .5))
                opt.close()

    def test_pair_motion_and_folded_knee(self):
        import json
        from s800_motion import split_leg_motion
        from experiment_utils import LEG_JOINT_TYPES
        names = json.loads((ROOT / 'scripts/pace/trajectory_plans/s800_legs_sim_20261012_22dof.json').read_text())['joint_order']
        for mode in LEG_JOINT_TYPES:
            center, amplitude, selected = split_leg_motion(names, mode)
            self.assertEqual(selected, select_leg_sweep(names, mode))
            self.assertEqual(len(selected), 2)
            self.assertEqual(torch.nonzero(amplitude).flatten().tolist(), selected)
        center, amplitude, selected = split_leg_motion(names, 'hip_pitch')
        torch.testing.assert_close(center[selected], torch.zeros(2))
        torch.testing.assert_close(amplitude[selected], torch.full((2,), .5))
        knees = select_leg_sweep(names, 'knee_pitch')
        torch.testing.assert_close(center[knees], torch.full((2,), 1.6))
        torch.testing.assert_close(amplitude[knees], torch.zeros(2))

    def test_expanded_bounds_preserve_frozen_joints_delay_and_original(self):
        # Two joints: fit only the second one. Check complete old domain is retained.
        bounds = torch.tensor([[.05, .5], [.005, .15], [0., 3.], [0., 1.],
                               [0., 4.15], [0., 1.6], [-.1, .1], [-.1, .1], [0., 10.]])
        original = bounds.clone()
        result = expand_fit_bounds(bounds, [1], 2.)
        torch.testing.assert_close(result[[1, 3, 5, 7]],
                                   torch.tensor([[.0025, .3], [0., 2.], [0., 3.2], [-.2, .2]]))
        torch.testing.assert_close(result[[0, 2, 4, 6, 8]], bounds[[0, 2, 4, 6, 8]])
        torch.testing.assert_close(bounds, original)
        self.assertTrue(torch.all(result[:, 0] <= bounds[:, 0]))
        self.assertTrue(torch.all(result[:, 1] >= bounds[:, 1]))
        torch.testing.assert_close(expand_fit_bounds(bounds, [1], 1.), bounds)
        for invalid in (.5, 0., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                expand_fit_bounds(bounds, [1], invalid)

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
            values = dict(plot_table=True, joint_order=names, selected=[0, 1], config={"gt": gt},
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

    def test_fixture_spec_and_contact_summary(self):
        from s800_fixture import (load_fixture_spec, default_fixture_path, part_world_pose,
                                  quat_from_rpy, summarize_contact, validate_fixture_spec,
                                  write_contact_csv, fixture_joint_limit_violation)
        spec = load_fixture_spec(default_fixture_path())
        self.assertTrue(spec["frame"].startswith("robot_base"))
        names = [part["name"] for part in spec["parts"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(part["shape"] in ("box", "cylinder") for part in spec["parts"]))
        parts_by_name = {part["name"]: part for part in spec["parts"]}
        self.assertTrue(parts_by_name["base_center"]["collision"])
        self.assertTrue(parts_by_name["base_rear"]["collision"])
        self.assertTrue(parts_by_name["column"]["collision"])
        # Base is entered relative to the measured pelvis frame (base at z=1.076).
        self.assertAlmostEqual(part_world_pose(spec["parts"][0], (0., 0., 1.076))[0][2], .095)
        np.testing.assert_allclose(quat_from_rpy([0., 0., np.pi / 2]),
                                   [np.sqrt(.5), 0., 0., np.sqrt(.5)], atol=1e-12)
        np.testing.assert_allclose(
            part_world_pose({"center": [1., 0., 0.], "rpy": [0., 0., 0.]}, (0., 0., 0.),
                            quat_from_rpy([0., 0., np.pi / 2]))[0],
            [0., 1., 0.], atol=1e-12)
        leg_names = ["J00_HIP_PITCH_L", "J01_HIP_ROLL_L",
                     "J06_HIP_PITCH_R", "J07_HIP_ROLL_R"]
        q = np.zeros((2, 4))
        self.assertTrue(fixture_joint_limit_violation(q, leg_names)["passed"])
        q[0, 0] = .01  # positive/backward hip pitch is blocked
        self.assertFalse(fixture_joint_limit_violation(q, leg_names)["passed"])
        q[0, 0] = 0.
        q[0, 1] = -.01  # left hip roll inward
        self.assertFalse(fixture_joint_limit_violation(q, leg_names)["passed"])
        q[0, 1] = 0.
        q[0, 3] = .01  # right hip roll inward
        self.assertFalse(fixture_joint_limit_violation(q, leg_names)["passed"])
        q[0, 3] = 0.
        q[1] = [-.1, .1, -.1, -.1]  # forward/outward only
        self.assertTrue(fixture_joint_limit_violation(q, leg_names)["passed"])
        q[0] = [.01, -.01, 0., 0.]
        self.assertTrue(fixture_joint_limit_violation(q, leg_names, margin=.02)["passed"])
        for bad in ({"parts": []},
                    {"parts": [{"name": "a", "shape": "box", "size": [1., 2.],
                                "center": [0., 0., 0.]}]},
                    {"parts": [{"name": "a", "shape": "nope", "center": [0., 0., 0.]}]},
                    {"parts": [{"name": "a", "shape": "box", "size": [1., -1., 1.],
                                "center": [0., 0., 0.]}]},
                    {"parts": [{"name": "a", "shape": "box", "size": [1., 1., 1.],
                                "center": [0., 0., 0.], "collision": "yes"}]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_fixture_spec(bad)
        force = np.zeros((5, 3))
        force[2, 1], force[3, 1], force[4, 0] = 2., 3., 1.5
        summary = summarize_contact(force, np.arange(5) * .005, ["base", "thigh", "shank"], 1.)
        self.assertTrue(summary["contact_detected"])
        self.assertAlmostEqual(summary["first_contact_time_s"], .010)
        self.assertEqual(summary["first_contact_body"], "thigh")
        self.assertAlmostEqual(summary["max_force_N"], 3.)
        self.assertAlmostEqual(summary["contact_duration_s"], .015)
        self.assertEqual(summary["peak_by_body_N"]["thigh"], 3.)
        base_only = np.zeros((5, 3))
        base_only[:, 0] = 100.
        ignored = summarize_contact(base_only, np.arange(5) * .005,
                                    ["base", "thigh", "shank"], 1., ignore_bodies=("base",))
        self.assertFalse(ignored["contact_detected"])
        self.assertEqual(ignored["max_force_N"], 0.)
        self.assertEqual(ignored["ignored_bodies"], ["base"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contact.csv"
            write_contact_csv(path, np.arange(5) * .005, force, ["base", "thigh", "shank"])
            self.assertEqual(len(path.read_text().splitlines()), 6)

    def test_leg_plan_preserves_center_and_rejects_crossed_legacy_sweep(self):
        from s800_clearance import LegClearance, with_midpoints
        from s800_motion import nominal_motion, choose_leg_motion
        path = ROOT / "source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/s800_pace_env_cfg.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
                    and isinstance(n.target, ast.Name) and n.target.id == "joint_order")
        names = ast.literal_eval(node.value)
        checker = LegClearance(ROOT / "assets_robot/engineai_S800/urdf/serial_s800_lock_wrist_torso.urdf", names)
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
        self.assertEqual([len(groups[g]) for g in ("legs", "torso", "arms")], [12, 0, 10])
        self.assertEqual(sorted(j for values in groups.values() for j in values), list(range(22)))
        with self.assertRaises(ValueError):
            select_groups("all,legs", names)
        # torso 在锁定 J12_TORSO_YAW 之后是空组：单独选它、或混在逗号列表里，
        # 都必须显式报错而不是静默跑一个不含上半身的实验
        with self.assertRaises(ValueError):
            select_groups("torso", names)
        with self.assertRaises(ValueError):
            select_groups("legs,torso", names)
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

    def test_real_delay_override_keeps_other_checks(self):
        names = ['joint']
        current = dict(physics_dt=.005, decimation=1, delay_mode='torque', kp=torch.ones(1))
        recorded = dict(current, delay_mode='command', runtime_simulation=False)
        data = dict(time=torch.arange(4)*.005, joint_names=names, dof_pos=torch.zeros(4,1),
                    des_dof_pos=torch.zeros(4,1), excited=names, experiment=recorded)
        with self.assertRaisesRegex(ValueError, 'delay_mode'):
            validate_recording(data, names, [0], current)
        validate_recording(data, names, [0], current, allow_real_delay_override=True)
        self.assertEqual(recorded['delay_mode'], 'command')
        with self.assertRaisesRegex(ValueError, 'kp'):
            validate_recording(data, names, [0], dict(current, kp=torch.zeros(1)), allow_real_delay_override=True)
        for flag in (True, None):
            recorded['runtime_simulation'] = flag
            with self.assertRaisesRegex(ValueError, 'delay_mode'):
                validate_recording(data, names, [0], current, allow_real_delay_override=True)
        recorded['runtime_simulation'] = False
        data['gt'] = {}
        with self.assertRaisesRegex(ValueError, 'delay_mode'):
            validate_recording(data, names, [0], current, allow_real_delay_override=True)

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
        data["experiment"].pop("actual_clearance")
        data["experiment"]["fixture"] = {"contact_detected": True}
        with self.assertRaisesRegex(ValueError, "fixture"):
            validate_recording(data, names, [0], current)
        validate_recording(data, names, [0], current, allow_fixture_contact=True)

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
