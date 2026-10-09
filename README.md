# ⚙️ PACE — Sim-to-Real Transfer for Legged Robots
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![Status](https://img.shields.io/badge/status-active_development-orange)

PACE is a framework for **sim-to-real transfer of diverse robotic systems**, combining data-driven system identification with evolutionary optimization.
It enables accurate actuator modeling and robust adaptation between simulation to reality by explicitly learning physically meaningful dynamics parameters.

---

## 🤖🐾 What is PACE?

PACE (Precise Adaptation through Continuous Evolution) bridges the gap between simulation and real hardware by:

* Estimating actuator and joint dynamics directly from measured data
* Using CMA-ES for parameter optimization
* Applying learned parameters to improve sim-to-real locomotion performance
* Supporting multiple robot platforms and actuator types

It is designed to integrate seamlessly with **NVIDIA Isaac Lab** and follows its task and environment conventions.

---

## 📦 Installation

### 1. Install Isaac Lab

Follow the official Isaac Lab installation guide:
[https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

We recommend using the **conda** or **uv** installation method, as this simplifies running Python scripts from the terminal and managing dependencies.

### 2. Clone this repository

Clone or copy this project separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
git clone https://github.com/leggedrobotics/pace-sim2real.git
cd pace-sim2real
```

### 3. Install PACE in editable mode

Using a Python interpreter that has Isaac Lab installed:

```bash
# Use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python'
# if Isaac Lab is not installed in a Python venv or conda environment
python -m pip install -e source/pace_sim2real
```
### ⚠️ Isaac Sim version compatibility

PACE is developed and tested against **Isaac Sim 5.0 / Isaac Lab (latest)**.

If you run the examples on older versions (e.g. Isaac Sim 4.5), you may encounter warnings such as:

```bash
Setting joint viscous friction coefficients are not supported in Isaac Sim < 5.0
```

These warnings are expected and indicate that certain physics properties (e.g. joint viscous friction) are not supported by the older simulator version. While the examples may still run, the physical fidelity and sim-to-real accuracy will be reduced.

✅ Recommended: Isaac Sim 5.0 or newer  
⚠️ Legacy support: May run with limitations on < 5.0

---

## 📚 Documentation

The full PACE documentation is available online:

👉 https://pace.filipbjelonic.com

This documentation site is built using **MkDocs** with the **Material for MkDocs** theme.

---

## 🐾 Running an Example: ANYmal D

### 1. Activate the Isaac Lab environment

```bash
conda activate <isaaclab_env>
cd path/to/pace-sim2real
```

### 1.5. Convert urdf into usd for training
convert urdf into usd using Isaaclab built-in toolbox
```bash
python scripts/tools/convert_urdf.py xxx.urdf xxx.usda --fix-base --merge-joints
```
xxx.urdf # path to urdf
xxx.usda # path to usd file
--fix-base # 固定机器人baselink 为了后续注入chirp信号
--merge-joints # 可选

### 2. Collect excitation data

(Alternatively, place your own real-world data in `data/`)

```bash
python scripts/pace/data_collection.py
```
if u are using s800 robot
```bash
python scripts/pace/data_collection.py --task=Isaac-Pace-S800-v0
```

if u are using Unitree G1 (29 dof)
```bash
python scripts/pace/data_collection.py --task=Isaac-Pace-G1-v0 --headless
```

This will collect simulation data and store results in:

```
data/<robot_name>/chirp_data.pt      # s800_sim / g1_sim / anymal_d_sim
```

#### 真值从哪来：单一数据源 `scripts/pace/robot_tables.py`

采集脚本按 env cfg 的 `robot_name` 从 **`scripts/pace/robot_tables.py`** 取该机器人的
「注入真值 + 激励设计」（armature / damping / friction / directions / bias / scale …）。
这份表同时被 `plot_trajectory.py` 和 `check_mirror_symmetry.py` import ——
**改真值只改这一个文件**，三处不会漂移。

⚠️ 几个硬约束（采集脚本会在注入前自动检查）：

- **真值必须落在 env cfg 的 `bounds_params` 内**。越界时 CMA-ES 会贴着边界收敛
  **且不报错**，表现是一个看起来正常的错值。
- **激励指令必须落在关节行程内**（`bias ± 0.4×半行程` 一般可以，但带重力塌陷的关节
  ——比如 G1 的 `waist_roll/pitch`——实际位置会被压低，要单独处理）。
- `directions` **每台机器人都不一样，不能照抄**，改完用
  `python scripts/check_mirror_symmetry.py --robot <s800|g1>` 校验（退出码 0 = 通过）。

采集完先做一次体检再往下走：关节有没有被锁死/贴限位、力矩和速度有没有顶到 limit、
**误差里高于扫频上限的能量占比**（超出指令带宽的频率只可能来自自激 —— 通常是
**延时 + 增益**导致的闭环失稳，不是激励设计的问题）。判据要用实际带宽：`--max_frequency`
现在默认 10Hz，所以看 >10Hz 的占比；对着 4Hz 的老数据（26_09_30）就得看 >4Hz。
⚠️ 别把带宽当常量记：26_09_29 采的是 0.1–2Hz、26_09_30 是 0.1–4Hz、G1 是 0.1–10Hz
（`chirp_data.pt` 的 `chirp` 字段记着实际值，老数据可由 `cma_es.infer_sweep_band()` 反解）。

### 3. Run PACE parameter fitting

```bash
python scripts/pace/fit.py
```

if u are using s800 robot
```bash
python scripts/pace/fit.py --headless --task=Isaac-Pace-S800-v0
```

if u are using Unitree G1
```bash
python scripts/pace/fit.py --headless --task=Isaac-Pace-G1-v0
```

This will estimate the actuator and joint parameters using CMA-ES and store results in:

```
logs/pace/<robot_name>/<YY_MM_DD_HH-MM-SS>/
```

⚠️ `--task` 的默认值是 `Isaac-Pace-S800-v0` —— 跑别的机器人**必须显式传**，否则会拿
S800 的 cfg 去找 `data/s800_sim/` 的数据，然后因为 `joint_order` 对不上而崩。

---

#### 3.1 参数布局（4n+1 维）

`4 × n_joints + 1`：S800 是 27 关节 → 109 维，G1 是 29 关节 → 117 维。

每一段内部按 env cfg 的 `joint_order` 排列（n = 27 时）：

| 下标 | 参数 | 单位 | 说明 |
|---|---|---|---|
| `[0:n]` | armature | kg·m² | 转子惯量，等效到关节侧 |
| `[n:2n]` | viscous friction | Nm·s/rad | 粘性摩擦 |
| `[2n:3n]` | coulomb friction | Nm | 库仑摩擦（PhysX 里 static = dynamic 时生效） |
| `[3n:4n]` | encoder bias | rad | 编码器零位偏差 |
| `[4n]` | delay | sim steps @400 Hz | 指令延时，全局一个 |

搜索区间来自各机器人 env cfg 的 `bounds_params`（`__post_init__` 里按关节类型设置），
**CMA-ES 的初始均值 = 区间中点**。⚠️ 注入的真值必须落在区间内 —— 见第 2 节的硬约束。

#### 3.2 命令行选项

| 选项 | 默认 | 说明 |
|---|---|---|
| `--task` | `Isaac-Pace-S800-v0` | 任务名，决定用哪份 env cfg。跑别的机器人必须显式传 |
| `--num_envs` | `4096` | 同时等于 CMA-ES 的种群大小 |
| `--headless` | off | 无 GUI（服务器上必加） |
| `--group` | `all` | 只优化某几组关节，见 3.3。可逗号组合 `legs,torso` |
| `--warm_start` | 无 | `mean_*.pt` 路径或 run 目录；冻结其余组并作为初始均值 |
| `--opt_delay` / `--no-opt_delay` | 开 | 每轮是否把全局 delay 一起放开重优化 |
| `--sigma` | cfg 的 `0.5` | CMA-ES 初始步长（归一化空间，盒子宽度为 2） |
| `--epsilon` | cfg 的 `1e-2` | 提前停止判据：种群 `(max-min)/min < epsilon` 即判收敛。设 `0` = 关掉 |

一次 `iteration` = 让全部 `num_envs` 个参数组合各跑完整条激励轨迹（8000 步 @400 Hz），
这是为什么一轮拟合要跑几小时。默认 `max_iteration=15000`；`epsilon=1e-2` 那个收敛判据
（种群离散度 <1%）实际几乎不会触发，所以：

- 正常跑完才会写最终 checkpoint；**中途 Ctrl-C 就取最新的 `mean_<iter>.pt`**（每 25 代存一次）。
- 已有两轮都是中断结束的，看到的 `mean_1975` / `mean_6450` 就是这么来的。

#### 3.3 分组拟合（手 / 腿 / 腰）

**为什么可以分组**：`serial_s800.urdf` 是三条挂在 `LINK_BASE` 上的独立支链（左腿 / 右腿 /
[腰→双臂]），而 fit 里 `LINK_BASE` 是焊死的（`fix_root_link=True`）→ 质量阵在支链之间
**块对角**，损失严格可加：

```
L = L_腿(θ_腿) + L_腰+臂(θ_腰, θ_臂)
```

即腿的轨迹只由腿的参数决定，**冻结的臂/腰取什么值都不影响腿的结果**（反之亦然）。
旁证：26_09_30 那轮腿错 400%、臂仍准到 0.1%，且臂的残差比腿小 9 倍 —— 真耦合的话
腿的错误会把臂的残差顶起来。

但**支链内部是耦合的**：`LINK_TORSO_YAW → J13/J27`，torso 是双臂的父关节，臂对 torso
也有反作用 → **torso 和 arms 必须同一轮优化**（`--group arms,torso`），不能先臂后腰。

**为什么要分组而不是一次搜 109 维**：各组参数的量纲差距很大 —— 腕关节的库仑摩擦占自身
力矩预算 ~5.8%，髋只有 0.04%（差 45 倍）。一套 bounds 尺度 + 一个 σ 覆盖全部 109 维时
必然牺牲一组：实测全场统一 bounds 那轮，轻关节准到 0.1%、髋/膝摩擦错 400%。

##### 流程 A：从零开始（不复用旧 run）

```bash
# 1) 腿：不给 --warm_start ⇒ 从中点全空间搜索（默认 σ=0.5）
python scripts/pace/fit.py --headless --task=Isaac-Pace-S800-v0 \
    --group legs --no-opt_delay --epsilon 0

# 2) 腰+臂：--warm_start 的作用只是把腿的收敛值带进输出文件。
#    臂/腰在 warm_start 里仍是中点，所以它们依然是从中点开始的全空间搜索，没有被热启动。
python scripts/pace/fit.py --headless --task=Isaac-Pace-S800-v0 \
    --group arms,torso --warm_start logs/pace/s800_sim/<上一轮目录> --no-opt_delay --epsilon 0

# 3) 收尾：此时各组都已收敛，全局跑一轮把 delay 定下来
python scripts/pace/fit.py --headless --task=Isaac-Pace-S800-v0 \
    --group all --warm_start logs/pace/s800_sim/<上一轮目录> --sigma 0.05
```

##### 流程 B：细化已有的全参数 run

```bash
# warm_start 指向已收敛的解 ⇒ 这是「接着细化」，σ 要调小
python scripts/pace/fit.py --headless --task=Isaac-Pace-S800-v0 \
    --group legs --warm_start logs/pace/s800_sim/26_09_29_12-41-22 --sigma 0.15
```

几点注意：

- **损失始终在全部 27 个关节上算**，不按组裁剪。在块对角的情形下这与按组裁剪等价，
  留着更简单；块内存在耦合（torso↔臂）时才必须保留。冻结组的参数值原样写进 `mean_*.pt`。
- `--warm_start` 有**两个作用**，用之前要分清想要哪个：① 冻结其余组的取值；② 作为本组
  CMA-ES 的初始均值。想做「从头搜」就**只**让它承担 ①（即让别的组一直是中点，见流程 A）；
  想「接着细化」就两者都要（见流程 B，记得配 `--sigma 0.1~0.15`）。
- **delay 是唯一的全局共享参数**，会把冻结组泄漏进本组：冻结点是错值时 `L_其他组` 对
  delay 的梯度也是错的。**分组阶段建议 `--no-opt_delay`，最后全局跑一轮时再定它。**
- **分数会偏高，且要关掉 `epsilon`**：冻结组的误差项对全种群是同一个常数 C
  （`score_i = S_本组(θ_i) + C`），加常数不改 argmin、排序不变，**优化本身不受影响**；
  但 `diff_score = (max-min)/min` 的分母被 C 抬高 → 判据被压低 → 可能把「还没收敛」
  误判成收敛而**提前结束**。分组轮加 **`--epsilon 0`** 关掉它。（`--group all` 时 C=0，
  判据不受影响，实测两轮全参数跑的 diff_score 一直在 1.4~30，从没接近过 1e-2。）
- 分组后维度降到 ~49，`--num_envs 4096` 是 84 倍冗余（损失是确定性的、没有噪声），
  降到 1024 能在同样时间里多跑 4 倍迭代。
- 冻结值若落在当前 bounds 外会被夹到边界并打印警告（切换 `UNIFORM_GT` 时要留意）。
- 还没做左右绑定（109→61）。S800 左右真值相同、激励按镜像设计，绑定可以零代价
  砍掉 L−R 方向的噪声，属于后续可选项。

#### 3.4 输出文件

```
logs/pace/<robot_name>/<YY_MM_DD_HH-MM-SS>/
├── config.pt            # bounds / joint_order / 本轮拟合用的 dof_pos,des_dof_pos,time
│                        #   + gt：**采集时注入的真值**（新数据文件才有）
├── mean_<iter>.pt       # 每 save_interval(=25) 代一次，4n+1 维物理量纲向量
├── best_trajectory.pt   # 该代最优个体的仿真轨迹 (T × n_joints)
└── events.out.tfevents.*  # TensorBoard
```

`mean_*.pt` 的布局与 3.1 的表一致，可以直接被 `--warm_start` 和 `plot_trajectory.py` 读取。

`config.pt` 里的 `gt` 是**这轮拟合用的数据在采集时注入的真值**（`data_collection.py` 写进
`chirp_data.pt`，fit 转发过来）。它让每个 run 自洽：`data/` 是共享路径、会被下一次采集
覆盖，出表时只能靠 run 自己的记录 —— 这也是 `plot_trajectory.py` 出表的首选真值来源。

#### 3.5 怎么判断一轮拟合好不好

**主指标是残差（轨迹预测误差），不是参数误差。** 参数误差存在一个由 `kp × RMS残差`
决定的分辨率下限：位置偏差 ≈ Δτ/kp，所以大 kp 关节（髋/膝 240）的摩擦误差被卡在
±(kp·残差) 量级，继续抠它收益很低。残差才是 sim2real 真正关心的量。

```bash
tensorboard --logdir logs/pace/s800_sim
```

TensorBoard 里按重要性看：

| tag | 含义 |
|---|---|
| `0_Episode/score` | **主指标**：该代最优个体的 Σ(位置误差²) 均值（27 关节、全时段） |
| `6_GroupResid/best_<组>` | 逐组 RMS 位置残差 [rad]，分组拟合的验收指标 |
| `0_Episode/diff_score` | 种群离散度 / 最优。持续在 ~10 以上说明还没收敛 |
| `5_BandLoss/share_*Hz` | 残差落在哪个频段：armature 效应 ∝ω² 住高频，摩擦与 ω 无关住低频 |
| `1_Armature` `2_Viscous_Friction` `3_Static_Dynamic_Friction` `4_Bias` | 逐关节参数收敛过程 |
| `0_Delay/best` | 延时收敛过程 |

分组拟合前的各组残差基线（S800，best member，同口径）：

| run | legs | torso | arms |
|---|---|---|---|
| `26_09_28_18-18-53` mean_1450 | 0.00316 | 0.00245 | 0.00297 |
| `26_09_29_12-41-22` mean_1975 | 0.00249 | 0.00262 | 0.00262 |
| `26_09_30_15-34-31` mean_6450（uniform GT） | 0.00852 | 0.00083 | 0.00094 |

判断标准：`--group legs` 跑完后 `6_GroupResid/best_legs` 应明显低于基线，且 torso/arms
不应变差。**如果没降就说明分组没用，不要继续跑 arms/torso。**

⚠️ 口径陷阱：`data_dir` 指向的 `chirp_data.pt` 会被下一次采集覆盖，所以每个 run 只能靠
自己的 `config.pt` 复现（真值也在里面，见 3.4）。老 run（26_09_28~30）的 config 里没有
`gt`，`plot_trajectory.py` 会回落到 `robot_tables.py` 并打印警告；其中 uniform 那轮要加
`--gt_preset uniform`，否则误差是拿分段真值算的假值。

### 4. Visualize results
```bash
python scripts/pace/plot_trajectory.py --plot_table --plot_trajectory --robot_name=s800_sim
# G1:
python scripts/pace/plot_trajectory.py --plot_table --robot_name=g1_sim
```
options
```bash
--plot_trajectory
--plot_table
--robot_name=s800_sim            # 也决定去哪找 run（logs/pace/<robot_name>/）
--folder_name=26_09_17_11-02-54  # 不给就取最新的 run
--mean_name=mean_499.pt          # 不给就取迭代数最大的 mean_*.pt
--gt_preset=auto                 # auto|design|uniform，见下
```

**表里的真值来源**（表头会自己打印出来）：

| 优先级 | 来源 | 适用 |
|---|---|---|
| 1 | run 的 `config.pt` 里的 `gt`（采集时注入的真值） | 新 run，**永远和数据本身一致** |
| 2 | `robot_tables.py` 里该机器人的表（回落） | 老 run（26_09_28~30） |
| 3 | `--gt_preset uniform` 的对照组 | 只对 S800 有意义（26_09_30 那轮） |

关节名对不上会**直接报错**，不会静默显示 0% 误差（那个「完美拟合」的假象踩过两次）。

配套校验脚本：

```bash
python scripts/check_mirror_symmetry.py --robot g1    # 激励镜像对称性（退出码 0 = 通过）
python scripts/joint_order_validation.py --task Isaac-Pace-G1-v0 --headless
                                                      # 关节名/驱动覆盖/限位/空动作 step
```

---

## 🧭 Usage

Recommended imports for custom projects:

```python
from pace_sim2real import PaceCfg, PaceSim2realEnvCfg, CMAESOptimizer
from pace_sim2real.utils import PaceDCMotorCfg, PaceDCMotor
import pace_sim2real.tasks  # registers environments
```

Please refer to the official documentation for further information.

⚠️ PACE is under active development. APIs may evolve as the framework matures.

---

## 📜 License

© 2025 ETH Zurich, Robotic Systems Lab, Filip Bjelonic

This project is licensed under the Apache License, Version 2.0.
See the LICENSE file for details.

---

## 👥 Maintainers

The PACE project is actively maintained by:

- Filip Bjelonic (ETH Zurich, RSL)
- René Zurbrügg (ETH Zurich, RSL)
- Oliver Fischer (ETH Zurich, RSL)

The maintainers are responsible for reviewing contributions, managing issues, and guiding the technical direction of the framework.

---

## 🙏 Acknowledgements

We would like to thank the RSL Learning Group for many insightful discussions that influenced the development of PACE. We are especially grateful to Konrad and Matthias for valuable technical input related to electronics and system integration.

We also acknowledge Zichong, Stephan, Efe, Yuntao, René, Clemens, Ryo, Alexander, and Fabio for employing and extending the PACE framework in their own research, providing valuable feedback and practical insights.

We further thank Oliver Fischer and René Zurbrügg for their direct contributions to the codebase and their ongoing support as project maintainers.

We also thank the early testers — Oliver, Clemens and Yasmine — who evaluated the framework prior to its public release and provided valuable feedback on usability, stability, and documentation improvements.

---

## 🛠 Code Formatting

We provide a pre-commit configuration to automatically format and lint the codebase.

### Install pre-commit

```bash
pip install pre-commit
```

### Enable hooks

```bash
pre-commit install
```

### Run manually

```bash
pre-commit run --all-files
```

---

## 🧩 Troubleshooting

### Pylance: Missing Indexing of Extensions

In some VS Code versions, indexing for Isaac Lab extensions may be incomplete. Add your extension path to `.vscode/settings.json`:

```json
{
  "python.analysis.extraPaths": [
    "<path-to-ext-repo>/source/pace_sim2real"
  ]
}
```

---

### Pylance: Crash / High Memory Usage

If Pylance crashes due to excessive indexing, exclude unused omniverse packages by commenting them out in:

`.vscode/settings.json` → `python.analysis.extraPaths`

Examples of packages that can often be safely excluded:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*",
"<path-to-isaac-sim>/extscache/omni.kit.*",
"<path-to-isaac-sim>/extscache/omni.graph.*",
"<path-to-isaac-sim>/extscache/omni.services.*"
```

---

## 🤝 Contributing & Feedback

Internal feedback is highly welcome during this phase.
Please report issues, unclear steps, or suggestions directly via GitHub issues or internal channels.

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute.

---

## 📖 How to cite

If you use **PACE Sim2Real** in your research, please cite our [paper](https://arxiv.org/pdf/2509.06342).

The paper has been accepted for publication in **The International Journal of Robotics Research (IJRR)**. The official IJRR citation will be added here once the article is published online.

> F. Bjelonic, F. Tischhauser, and M. Hutter,  
> _Towards Bridging the Gap: Systematic Sim-to-Real Transfer for Diverse Legged Robots_, arXiv:2509.06342, 2025.

```bibtex
@article{bjelonic2025towards,
  title         = {Towards Bridging the Gap: Systematic Sim-to-Real Transfer for Diverse Legged Robots},
  author        = {Bjelonic, Filip and Tischhauser, Fabian and Hutter, Marco},
  journal       = {arXiv preprint arXiv:2509.06342},
  year          = {2025},
  eprint        = {2509.06342},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
}
```

---

## ⭐ Star History

[![PACE Star History](https://raw.githubusercontent.com/leggedrobotics/pace-sim2real/star-history-data/star-history.svg)](
  https://pace.filipbjelonic.com/star-history/
)

---

**PACE** — bringing simulation and reality closer, one parameter at a time.
