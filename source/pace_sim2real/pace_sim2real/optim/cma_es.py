# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

import cmaes
import math
import torch
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
from datetime import datetime
import os

class CMAESOptimizer:
    def __init__(self, bounds, population_size, log_dir, joint_order, max_iteration, data, device, epsilon=None, sigma=0.5, save_interval=10, save_optimization_process=False, segment_edges_hz=(0.1, 4.0), sweep_kind="linear", groups=None, active_groups=None, warm_start=None, opt_delay=True, loss_joints=None, seed=0):

        if max_iteration < 1 or population_size < 2:
            raise ValueError("Need max_iteration>=1 and population_size>=2")
        self.joint_order = joint_order
        self.max_iteration = max_iteration
        self.epsilon = epsilon
        self.save_interval = save_interval
        self.device = device
        self.save_optimization_process = save_optimization_process
        self._timer_start = datetime.now()  # timer for logging purposes

        # create log_dir in YY_MM_DD_hh-mm-ss format
        folder_time = datetime.now().strftime("%y_%m_%d_%H-%M-%S")
        # create string with time and date
        log_dir = os.path.join(log_dir, folder_time)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = TensorboardSummaryWriter(log_dir=log_dir)
        torch.save({**data, "bounds": bounds,
                    "joint_order": joint_order,
                    "dof_pos": data["dof_pos"],
                    "des_dof_pos": data["des_dof_pos"],
                    "time": data["time"]
                    }, log_dir + "/config.pt")

        self.bounds = bounds
        # CMA-ES 实例在这里不建 —— 挪到 delay_idx 之后，分组优化要先按参数块算出
        # active 子集，只在子空间里建优化器。

        self.scores_counter = 0
        self.iteration_counter = 0

        self.scores = torch.zeros(population_size, device=device)
        self.scores_buffer = torch.zeros((max_iteration, population_size), device=device)
        self.sim_dof_pos_buffer = torch.zeros((population_size, data["dof_pos"].shape[0], len(joint_order)), device=device)

        self.params = torch.zeros((population_size, bounds.shape[0]), device=device)
        self.sim_params = torch.zeros_like(self.params)
        if save_optimization_process:
            self.sim_params_buffer = torch.zeros((max_iteration, population_size, bounds.shape[0]), device=device)

        num_joints = len(joint_order)
        self.armature_idx = slice(0, num_joints)
        self.damping_idx = slice(num_joints, 2 * num_joints)
        self.friction_idx = slice(2 * num_joints, 3 * num_joints)
        self.bias_idx = slice(3 * num_joints, 4 * num_joints)
        self.delay_idx = 4 * num_joints

        self.loss_joints = list(range(num_joints)) if loss_joints is None else list(loss_joints)
        if not self.loss_joints or min(self.loss_joints) < 0 or max(self.loss_joints) >= num_joints:
            raise ValueError("Invalid loss joint indices")
        # Mask losses as well as parameters; unexcited joints still evolve physically.
        self.groups = {k: [int(j) for j in v] for k, v in
                       (groups if groups else {"all": list(range(num_joints))}).items()}
        self.group_idx = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in self.groups.items()}
        self.real_dof_pos = data["dof_pos"].to(device)

        _blocks = (0, num_joints, 2 * num_joints, 3 * num_joints)  # armature / damping / friction / bias
        if active_groups is None or "all" in active_groups:
            _act_joints = sorted({j for v in self.groups.values() for j in v})
        else:
            _missing = set(active_groups) - set(self.groups)
            if _missing:
                raise ValueError(f"未知分组 {sorted(_missing)}，可用: {sorted(self.groups)}")
            _act_joints = sorted({j for g in active_groups for j in self.groups[g]})
        _act = sorted({j + off for j in _act_joints for off in _blocks})
        if opt_delay:
            _act.append(self.delay_idx)          # 延时是全局的，每一轮都放开重新定
        self.active_idx = torch.tensor(_act, dtype=torch.long, device=device)
        self.opt_delay = opt_delay

        # 冻结维的参考值（归一化空间 [0,1] 之外无意义，一律夹到 [-1,1]）。
        # warm_start 给的是物理量纲的完整 109 维向量（就是 save_checkpoint 写的 mean_*.pt）。
        if warm_start is not None:
            _ws = torch.as_tensor(warm_start, device=device).float().flatten()
            if _ws.numel() != bounds.shape[0]:
                raise ValueError(f"warm_start 长度 {_ws.numel()} != 参数个数 {bounds.shape[0]}")
            _z = 2.0 * (_ws - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) - 1.0
            _oob = (_z < -1.0) | (_z > 1.0)
            if _oob.any():
                _blk_names = ("armature", "damping", "friction", "bias", "delay")
                _desc = []
                for _i in _oob.nonzero().flatten().tolist():
                    _b = _i // num_joints if _i < 4 * num_joints else 4
                    _jn = joint_order[_i % num_joints] if _i < 4 * num_joints else "delay"
                    _desc.append(f"{_blk_names[_b]}@{_jn}={_ws[_i].item():.4g}")
                print(f"[group] ⚠️ warm_start 有 {int(_oob.sum())} 个值落在 bounds 外，已夹到边界："
                      f"{', '.join(_desc[:8])}{' ...' if len(_desc) > 8 else ''}")
                print("[group] ⚠️ 这些维度若是冻结维，结果里的值就是被夹过的，不是 warm_start 原值。")
            self.z_ref = _z.clamp(-1.0, 1.0)
        else:
            self.z_ref = torch.zeros(bounds.shape[0], device=device)

        # 只在 active 子空间里建 CMA-ES；初始 mean = warm_start 在这几维上的取值
        # （没有 warm_start 就是 0 = bounds 中点，与分组前的行为一致）。
        _sub = bounds[self.active_idx]
        _sub_norm = torch.ones_like(_sub)
        _sub_norm[:, 0] *= -1
        self.optimizer = cmaes.CMA(mean=self.z_ref[self.active_idx].cpu().numpy(), sigma=sigma,
                                   bounds=_sub_norm.cpu().numpy(), seed=seed, population_size=population_size)
        _frozen_n = bounds.shape[0] - len(_act)
        print(f"[group] active={len(_act)} 维 / {len(_act_joints)} 关节"
              f"（{'含' if opt_delay else '不含'} delay），冻结 {_frozen_n} 维")
        _opt_names = ([k for k, v in self.groups.items() if set(v) <= set(_act_joints)]
                      if active_groups is not None else ["全部"])
        print(f"[group] 初始 mean 取自 {'warm_start' if warm_start is not None else 'bounds 中点'}；"
              f"优化组={_opt_names}")

        # --- 分段损失诊断：只写 TB，不改变 CMA-ES 看到的目标 -------------------
        # 线性 chirp 的瞬时频率是已知的：f(t) = f0 + (f1-f0)·t/T，所以按时间切段
        # 等价于按频率切段。用途是看残差落在哪个频段，从而判断哪个参数没被约束住：
        #   armature 效应 ∝ ω²     → 住高频
        #   friction  τf·sign(q̇)   → 与 ω 无关，相对惯性项在低频占比最大
        # 26_09_29 那轮的实测：1.6-2.0Hz 一段占损失的 74%，0.1-0.8Hz 三档合计仅 5%
        # —— 正好对应 friction 误差 7.7% 远大于 armature 误差 1.2%。
        # ⚠️ 首尾边界必须等于 data_collection.py 的 --min_frequency/--max_frequency，
        #    否则「段号 → 频率」的映射是错的。
        self.seg_edges_hz = tuple(segment_edges_hz)
        _n = data["dof_pos"].shape[0]
        # 先查长度再取下标：空 tuple / 单元素都用来关闭诊断
        if len(self.seg_edges_hz) >= 2 and self.seg_edges_hz[-1] > self.seg_edges_hz[0]:
            _f_lo, _f_hi = self.seg_edges_hz[0], self.seg_edges_hz[-1]
            # 「时间步 ↔ 瞬时频率」的映射取决于扫频方式，必须和 data_collection.py 一致，
            # 否则段号对应的频段是错的（不报错，只是标签错）。
            if sweep_kind == "log":
                # 对数扫频: f(t) = f0·(f1/f0)^(t/T)  →  t/T = ln(f/f0)/ln(f1/f0)
                _at = lambda f: _n * math.log(f / _f_lo) / math.log(_f_hi / _f_lo)
            else:
                # 线性扫频: f(t) = f0 + (f1-f0)·t/T
                _at = lambda f: _n * (f - _f_lo) / (_f_hi - _f_lo)
            self.seg_edges = [min(_n, max(0, int(round(_at(f))))) for f in self.seg_edges_hz]
            self.seg_scores = torch.zeros((population_size, len(self.seg_edges_hz) - 1), device=device)
        else:
            self.seg_edges = None
            self.seg_scores = None

        self._reset_population()
        print("CMA-ES optimizer initialized.")
        print("Current iteration: ", self.iteration_counter)

    def ask(self):
        return self.optimizer.ask()

    def tell(self, sim_dof_pos, real_dof_pos):
        residual = sim_dof_pos - real_dof_pos - self.sim_params[:, self.bias_idx]
        err_sq = torch.sum(residual[:, self.loss_joints].square(), dim=1)
        self.scores += err_sq
        # 同步累加分段损失。纯诊断 —— CMA-ES 只看到上面的 self.scores。
        if self.seg_scores is not None:
            self.seg_scores[:, self._seg_of(self.scores_counter)] += err_sq
        self.sim_dof_pos_buffer[:, self.scores_counter, :] = sim_dof_pos
        self.scores_counter += 1

    def _seg_of(self, step):
        """时间步 → chirp 频段号。线性 chirp 下时间步与瞬时频率一一对应。"""
        seg = 0
        for i, edge in enumerate(self.seg_edges):
            if step >= edge:
                seg = i
        return min(seg, self.seg_scores.shape[1] - 1)

    def evolve(self):
        self.scores /= self.scores_counter
        self.scores_buffer[self.iteration_counter, :] = self.scores
        if self.save_optimization_process:
            self.sim_params_buffer[self.iteration_counter, :, :] = self.sim_params
        best = int(torch.argmin(self.scores))
        self._last_best_params = self.sim_params[best].detach().cpu().clone()
        self._last_best_trajectory = self.sim_dof_pos_buffer[best].detach().cpu().clone()
        solutions = []
        for i in range(self.optimizer.population_size):
            # 只把 active 子空间交给 CMA-ES（冻结维不进 tell）
            solutions.append((self.params[i, self.active_idx].cpu().numpy(), self.scores[i].item()))
        self.optimizer.tell(solutions)
        if self.save_interval > 0 and self.iteration_counter % self.save_interval == 0:
            self.save_checkpoint(self._params_to_sim_params(self._sub_to_full(self.optimizer._mean)), self.iteration_counter)
        self._print_iteration()

        self._reset_population()

        self.scores = torch.zeros_like(self.scores)
        self.scores_counter = 0
        if self.seg_scores is not None:
            self.seg_scores.zero_()
        self.iteration_counter += 1
        print("CMA-ES optimizer iteration: ", self.iteration_counter)

    def finished(self):
        finished = self.max_iteration <= self.iteration_counter
        diff_score = (self.scores_buffer[self.iteration_counter - 1, :].max() - self.scores_buffer[self.iteration_counter - 1, :].min()) / self.scores_buffer[self.iteration_counter - 1, :].min()
        # bool() 不能省：diff_score 是 0-dim tensor，or 会把 tensor 直接透传出来，
        # 调用方拿到的就不是 bool 了（fit.py 里靠 if 的真值判断侥幸能用）。
        # 注意 epsilon=0 时 diff_score < 0 恒不成立 ⇒ 等于关掉这个提前停止判据。
        finished = bool(finished or (self.epsilon is not None and diff_score < self.epsilon))
        if finished:
            print("CMA-ES optimization finished.")
            self.save_checkpoint(self._params_to_sim_params(self._sub_to_full(self.optimizer._mean)), self.iteration_counter - 1, finished=True)
        return finished

    def _reset_population(self):
        # 冻结维固定为 z_ref，只有 active_idx 维来自 CMA-ES 采样
        self.params[:] = self.z_ref.unsqueeze(0)
        for i in range(self.optimizer.population_size):
            # dtype 必须显式指定：ask() 是 float64 numpy，而花式索引赋值（index_put_）
            # 不像基本切片那样隐式转 dtype，不加会 RuntimeError。
            self.params[i, self.active_idx] = torch.as_tensor(
                self.optimizer.ask(), device=self.device, dtype=self.params.dtype)
        self.sim_params = self._params_to_sim_params(self.params)

    def _sub_to_full(self, sub_mean):
        """把子空间里的 CMA-ES 均值展开回完整参数向量（冻结维取 z_ref）。"""
        z = self.z_ref.clone()
        z[self.active_idx] = torch.as_tensor(sub_mean, device=self.device, dtype=z.dtype).flatten()
        return z

    def _group_resid(self, min_index):
        """最优个体在每组关节上的 RMS 位置残差 [rad] —— 分组优化的验收指标。

        与 TB 的 0_Episode/score 同源（都是 sim - real - bias），只是按组分摊并
        除以关节数，所以组间、轮次间可比。用整个轨迹重算，不复用 tell() 里的累加。
        """
        resid = self.sim_dof_pos_buffer[min_index] - self.real_dof_pos - self.sim_params[min_index, self.bias_idx]
        return {g: resid[:, idx].pow(2).mean().sqrt().item() for g, idx in self.group_idx.items()}

    def update_simulator(self, articulation, joint_ids, initial_position):
        env_ids = torch.arange(len(self.sim_params[:, self.armature_idx]), device=self.device, dtype=torch.int32)
        articulation.write_joint_armature_to_sim(self.sim_params[:, self.armature_idx], joint_ids=joint_ids, env_ids=env_ids)
        articulation.data.joint_armature[:, joint_ids] = self.sim_params[:, self.armature_idx]
        articulation.write_joint_viscous_friction_coefficient_to_sim(self.sim_params[:, self.damping_idx], joint_ids=joint_ids, env_ids=env_ids)
        articulation.data.joint_viscous_friction_coeff[:, joint_ids] = self.sim_params[:, self.damping_idx]
        # If we set static friction lower than dynamic friction, the sim complains. So we need to do this weird order.
        # articulation.write_joint_dynamic_friction_coefficient_to_sim(0.0, joint_ids=joint_ids, env_ids=env_ids)
        articulation.write_joint_dynamic_friction_coefficient_to_sim(
            torch.zeros((env_ids.shape[0], joint_ids.shape[0]), device=self.device, dtype=torch.float32),
            joint_ids=joint_ids,
            env_ids=env_ids
        )
        articulation.write_joint_friction_coefficient_to_sim(self.sim_params[:, self.friction_idx], joint_ids=joint_ids, env_ids=env_ids)
        articulation.data.joint_friction_coeff[:, joint_ids] = self.sim_params[:, self.friction_idx]
        articulation.write_joint_dynamic_friction_coefficient_to_sim(self.sim_params[:, self.friction_idx], joint_ids=joint_ids, env_ids=env_ids)
        articulation.data.joint_dynamic_friction_coeff[:, joint_ids] = self.sim_params[:, self.friction_idx]
        articulation.write_joint_position_to_sim(initial_position + self.sim_params[:, self.bias_idx], joint_ids=joint_ids)
        articulation.write_joint_velocity_to_sim(torch.zeros_like(initial_position), joint_ids=joint_ids)
        for drive_type in articulation.actuators.keys():
            drive_indices = articulation.actuators[drive_type].joint_indices
            if isinstance(drive_indices, slice):
                all_idx = torch.arange(joint_ids.shape[0], device=joint_ids.device, dtype=torch.int32)
                drive_indices = all_idx[drive_indices]
            comparison_matrix = (joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0))
            drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
            articulation.actuators[drive_type].update_encoder_bias(self.sim_params[:, self.bias_idx][:, drive_joint_idx])
            # ⚠️ 必须 round 而不是直接 to(torch.int)（那是截断）：延时是整数，目标函数对
            #    这个连续维是阶梯状的，截断会让「真值 N」落在格子边界上（θ∈[N,N+1) 才是 N），
            #    于是初值/真值一有微小扰动就掉到 N-1。floor test 把 bounds 收缩到真值附近时
            #    会因此劈成两半（G1 实测：一半样本延时差 1 步，损失差 9 个数量级）。
            #    取整后真值落在格子中心 [N-0.5, N+0.5)，框收缩不跨界、搜索也不贴边界。
            articulation.actuators[drive_type].update_time_lags(
                torch.round(self.sim_params[:, self.delay_idx]).to(torch.int))
            articulation.actuators[drive_type].reset(env_ids)

    def _print_iteration(self):
        min_score = torch.min(self.scores)
        max_score = torch.max(self.scores)
        min_index = torch.argmin(self.scores)
        print("Max score: ", max_score.item())
        print("Min score: ", min_score.item(), " at index: ", min_index.item())
        print("Armature: ", self.sim_params[min_index, self.armature_idx].tolist())
        print("Viscous Friction: ", self.sim_params[min_index, self.damping_idx].tolist())
        print("Static/Dynamic Friction: ", self.sim_params[min_index, self.friction_idx].tolist())
        print("Bias: ", self.sim_params[min_index, self.bias_idx].tolist())
        print("Delay: ", self.sim_params[min_index, self.delay_idx].tolist())
        # 分组残差：这一轮最优个体在各组关节上的 RMS 位置残差 [rad]。
        # 判断「分组有没有用」看这个数降没降，不要看参数误差。
        self._group_resid_last = self._group_resid(min_index.item())
        print("Group RMS resid [rad]: " + ", ".join(f"{g}={v:.5f}" for g, v in self._group_resid_last.items()))
        print(f"Elapsed time: {(datetime.now() - self._timer_start).total_seconds():.1f} seconds")
        self._timer_start = datetime.now()
        self._log()

    def _params_to_sim_params(self, params):
        sim_params = (params + 1.0) / 2.0  # change range from 0 to 1
        sim_params = self.bounds[:, 0] + sim_params * (self.bounds[:, 1] - self.bounds[:, 0])  # range from lower to upper bound
        return sim_params

    def get_best_sim_params(self):
        best_params = self._sub_to_full(self.optimizer._mean)
        return self._params_to_sim_params(best_params)

    def _log(self):
        min_score, min_score_index = torch.min(self.scores, dim=0)
        max_score, _ = torch.max(self.scores, dim=0)
        for i in range(len(self.joint_order)):
            self.writer.add_histogram("4_Bias/distribution_" + self.joint_order[i], self.sim_params[:, self.bias_idx][:, i], self.iteration_counter)
            self.writer.add_histogram("3_Static_Dynamic_Friction/distribution_" + self.joint_order[i], self.sim_params[:, self.friction_idx][:, i], self.iteration_counter)
            self.writer.add_histogram("2_Viscous_Friction/distribution_" + self.joint_order[i], self.sim_params[:, self.damping_idx][:, i], self.iteration_counter)
            self.writer.add_histogram("1_Armature/distribution_" + self.joint_order[i], self.sim_params[:, self.armature_idx][:, i], self.iteration_counter)

            self.writer.add_scalar("4_Bias/best_" + self.joint_order[i], self.sim_params[min_score_index, self.bias_idx][i].item(), self.iteration_counter)
            self.writer.add_scalar("3_Static_Dynamic_Friction/best_" + self.joint_order[i], self.sim_params[min_score_index, self.friction_idx][i].item(), self.iteration_counter)
            self.writer.add_scalar("2_Viscous_Friction/best_" + self.joint_order[i], self.sim_params[min_score_index, self.damping_idx][i].item(), self.iteration_counter)
            self.writer.add_scalar("1_Armature/best_" + self.joint_order[i], self.sim_params[min_score_index, self.armature_idx][i].item(), self.iteration_counter)
        self.writer.add_histogram("0_Delay/distribution", self.sim_params[:, self.delay_idx], self.iteration_counter)
        self.writer.add_scalar("0_Delay/best", self.sim_params[min_score_index, self.delay_idx].item(), self.iteration_counter)

        self.writer.add_scalar("0_Episode/score", min_score.item(), self.iteration_counter)
        self.writer.add_scalar("0_Episode/max_score", max_score.item(), self.iteration_counter)
        self.writer.add_scalar("0_Episode/diff_score", (max_score - min_score) / min_score, self.iteration_counter)

        # 分组残差（弧度 RMS，按组内关节数摊平）：分组优化的验收指标。
        # 跨轮次可比 —— 对比「分组前基线」和「分组后」看这几条线降没降。
        for _g, _v in getattr(self, "_group_resid_last", {}).items():
            self.writer.add_scalar(f"6_GroupResid/best_{_g}", _v, self.iteration_counter)

        # 分段损失：残差的频段分布。mean_* 与 0_Episode/score 同尺度；
        # share_* 是该段占总损失的份额（若干份额之和恒为 1），份额最高的那段
        # 就是优化器真正在优化的频段 —— 哪个参数没被约束住，看它住在哪一段。
        if self.seg_scores is not None:
            n_steps = torch.tensor(
                [max(1, self.seg_edges[s + 1] - self.seg_edges[s]) for s in range(self.seg_scores.shape[1])],
                device=self.device,
                dtype=torch.float32,
            )
            band_mean = self.seg_scores / n_steps
            band_share = self.seg_scores / self.seg_scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
            for s in range(self.seg_scores.shape[1]):
                tag = f"{self.seg_edges_hz[s]:.1f}-{self.seg_edges_hz[s + 1]:.1f}Hz"
                self.writer.add_scalar("5_BandLoss/mean_" + tag, band_mean[:, s].mean().item(), self.iteration_counter)
                self.writer.add_scalar("5_BandLoss/share_" + tag, band_share[:, s].mean().item(), self.iteration_counter)

    def save_checkpoint(self, mean, iteration, finished=False):
        best_traj = self._last_best_trajectory
        torch.save(self._last_best_params, os.path.join(self.writer.log_dir, "best_params.pt"))
        torch.save(best_traj, os.path.join(self.writer.log_dir, "best_trajectory.pt"))
        torch.save(mean, os.path.join(self.writer.log_dir, "mean_" + f"{iteration:03}" + ".pt"))
        if finished and self.save_optimization_process:
            torch.save({"params_buffer": self.sim_params_buffer,
                        "scores_buffer": self.scores_buffer, },
                       os.path.join(self.writer.log_dir, "progress.pt"))

    def close(self):
        self.writer.close()
