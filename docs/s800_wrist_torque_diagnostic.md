# S800：26 自由度 wrist 保持力矩诊断

## 2026-11-05 已接受的仿真基线

最新 `chirp_arms_wrist26_shoulder80_kp40_kd7.pt` 的元数据实际为 **Kp=20、Kd=1**（文件名沿用了旧值），0.1–4 Hz、20 秒 chirp、前后各 1 秒 rest。Wrist pitch 请求峰值约 4.6529 N·m，roll 约 3.2566 N·m，均未超过本地 URDF 的 5 N·m 参考线。这不等于真机安全认证。

`s800_motion.py` 已将肩 pitch/roll/yaw 幅值 .28/.24/.20 rad、肘 pitch/yaw .30/.15 rad、腕零位零幅值记为当前手臂基线。`data_collection.py` 的 wrist 诊断默认增益更新为 20/1。下文 100/7 的显式命令保留作为历史对照，并非当前默认。

原有腿部支架轨迹保持不变；若要精确复现这次测试的整机目标（包括历史腿部保持姿态），继续使用 `--trajectory_plan scripts/pace/trajectory_plans/s800_sim_20261012_wrist26_shoulder80.json`。无 trajectory_plan 时使用 s800_motion 当前配置，`--trajectory_profile legacy` 可避免附加缩放。

本流程只用于仿真预检查。默认 22 自由度环境和 fit 不变；仅 `data_collection.py --wrist_diagnostic` 切换到新资产。Isaac Sim 转换和仿真由用户手动执行。

## 1. URDF → USDA

已生成 `assets_robot/robot_s/urdf/serial_robot_s_pace_26dof.urdf`。
以当前 22 自由度 URDF 为基础，只将 J18/J19/J32/J33 改回 revolute，并对照完整 URDF 验证四个关节定义。躯干、头和手指仍 fixed，link 质量、惯量、几何均未改，总质量仍为 83.78593860 kg。

在 `env_isaaclab` 环境、项目根目录执行转换：

```bash
cd /home/samsung/Humanoid/pace-sim2real
LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}" \
python /home/samsung/Humanoid/IsaacLab/scripts/tools/convert_urdf.py \
  assets_robot/robot_s/urdf/serial_robot_s_pace_26dof.urdf \
  assets_robot/robot_s/usd_wrist_diagnostic/serial_robot_s_pace_26dof.usda \
  --fix-base --merge-joints --headless
```

若以后需要重新生成 URDF（此步不启动仿真）：

```bash
python scripts/pace/wrist_diagnostic.py
```

使用独立 USD 目录，避免转换器覆盖旧资产附属文件。采集启动时会校验 USD 的可动关节集合必须恰好等于指定的 26 个关节。

## 2. 手臂扫频，wrist 仅保持零位

```bash
python -u scripts/pace/data_collection.py \
  --group arms --wrist_diagnostic \
  --wrist_kp 100 --wrist_kd 7 --encoder_bias 0 \
  --trajectory_profile legacy \
  --delay_mode command --delay_steps 0 \
  --duration 20 --min_frequency 0.1 --max_frequency 4 --rest 1 --ramp 2 \
  --output s800_sim/chirp_arms_wrist26_hold.pt
```

这里有 10 个肩/肘关节被激励，四个 wrist 目标始终为 0，幅值为 0；全部 26 个关节均采集。共 22 秒（1 秒 rest + 20 秒 chirp + 1 秒 rest），控制/物理步长均为 0.005 秒。

`Kp=100、Kd=7、目标=0` 来自此前真机手臂记录 `data/s800_real/chirp_1793777886655542299_1273_arm/experiment.yaml` 的 `full_kp/full_kd/hold_position`，不是 kpkd_params 中的参考增益。这是复现既有设置的基线，不是推荐真机增益。

`delay_steps=0` 是无人工延迟的诊断基线，不表示真机没有延迟。之后可单独改变该参数作敏感性对比。`legacy` 保持 s800_motion.py 当前中心和幅值，不额外缩放。

观察较低保持增益时，用另一输出文件，其他参数保持一致，例如：

```bash
python -u scripts/pace/data_collection.py \
  --group arms --wrist_diagnostic \
  --wrist_kp 12.5 --wrist_kd 0.13 --encoder_bias 0 \
  --trajectory_profile legacy \
  --delay_mode command --delay_steps 0 \
  --duration 20 --min_frequency 0.1 --max_frequency 4 --rest 1 --ramp 2 \
  --output s800_sim/chirp_arms_wrist26_hold_kp12p5_kd0p13.pt
```

后一组仅为 kpkd_params 参考增益的对照，不能只用力矩下降判断更好，还需看 wrist 偏离零位的幅度和振荡。

## 3. 输出与解释

### 恢复 2026-10-10～12 仿真采集时的轨迹

当前 s800_motion.py 的肩肘幅值经过真机测试缩小，为该历史版本的 1/3。`legacy` 表示使用当前参数，不表示历史参数。要恢复历史中心和幅值，在上述采集命令中额外指定：

```bash
--trajectory_plan scripts/pace/trajectory_plans/s800_sim_20261012_wrist26.json
```

同时使用独立输出名 `--output s800_sim/chirp_arms_wrist26_sim_original.pt`。该计划来自保存的 2026-10-11 仿真轨迹，与代码提交 f2532ff（2026-10-12）一致。肩 pitch/roll/yaw、肘 pitch/yaw 的幅值绝对值依次为 .35/.30/.25/.30/.15 rad，原 22 个关节的中心姿态也一并恢复；新增 wrist 仍为零位、零幅值。已离线验证原 22 个关节的全部 4400 个目标样本与历史配置完全一致。

这仅恢复轨迹参数，USD、动力学假设、PD、编码器偏置、延迟仍由本次诊断配置决定，不是复现整个历史实验。

仍显示原来的三张图：位置、力矩、速度。诊断模式同时保存同名 `.position.png`、`.torque.png`、`.velocity.png`；`--no-plot` 会关闭绘图，但仍保存 PT 和 CSV。

输出文件位于 `data/s800_sim/`：

- `chirp_arms_wrist26_hold.pt`：完整状态、目标、力矩、模型参数与诊断元数据。
- `chirp_arms_wrist26_hold.trajectory.json`：实际使用的中心和幅值。
- `chirp_arms_wrist26_hold.torque.csv`：每个周期、每个关节的三种力矩。
- `chirp_arms_wrist26_hold.torque_summary.csv`：逐关节峰值、参考阈值、超限比例和首次超限时间；也会打印到终端。

| 字段 | 含义 |
| --- | --- |
| `pd_requested` | 用本行采样的编码器位置、速度与未延迟目标计算 `kp*(q_des-q)-kd*qd`，与 SDK 发布前检查公式一致；不做裁剪 |
| `pd_computed` | 本周期实际进入执行器的目标产生的 PD 力矩，限幅之前；command delay 模式下包含目标延迟 |
| `applied_torque_step` / CSV `applied` | 执行器返回的施加力矩，包含限幅和所选延迟行为 |
| `dof_torque` | 为兼容旧采集数据保留的步进前读取值，属于上一个周期；本次诊断使用上面三个新字段 |

位置/速度在 env.step 前采样；computed/applied 在该次 env.step 后读取缓存。decimation=1，因此不会漏掉中间控制周期。torque delay 模式下 computed 与 applied 对应不同队列时刻的命令，不能直接将两者差值全部解释为限幅。

当前 URDF 四个 wrist 的 effort 为 **5 N·m**。红色参考线和 `reference_exceed_percent` 使用本地 URDF effort；`computed_over_sim_limit_percent` 使用仿真执行器 effort_limit。SDK 的实际保护使用运行时 `GetTorqueLimit`，这里没有读取真机运行时阈值，不能将 URDF 参考线当作已核实的 SDK 阈值。

## 模型边界

- Wrist 的 armature=.005 kg·m²、粘滞摩擦=.1 N·m·s/rad、库仑摩擦=.05 N·m，是显式临时假设；其余关节仍使用 data_collection 的人工 sim-to-sim GT。全部写入 PT，不是真机辨识结果。
- 诊断默认编码器偏置 0；普通采集仍为 .05。可显式传 `--encoder_bias`。
- 仿真初始状态直接放到目标姿态，未复现真机的 5 秒入场与 2 秒增益切换。该结果仅覆盖 rest/chirp/rest。
- 支架未建模，躯干/头/手指仍机械锁定；给出的命令延续分组采集默认关闭自碰撞，可追加 `--self_collisions` 单独观察自碰撞影响。
- 力矩是执行器输出，不是关节约束反力或碰撞力矩；原图 `net_torque` 是按速度估算摩擦后相减，不是传感器测得的净关节力矩。
- 输出超过阈值时仿真继续记录，便于定位完整曲线；真实 SDK 会停止。不能据此判断真机允许继续运行。
- 26 自由度诊断数据不能直接用于现有 22 自由度 fit，也不要与历史 22 自由度 checkpoint 混用。

离线检查已覆盖：URDF 仅改四个关节类型、26 关节映射与旧 bounds 保留、腕部零激励、原始 PD 超限但施加力矩被裁剪的情况。未运行 USD 转换或动态仿真。
