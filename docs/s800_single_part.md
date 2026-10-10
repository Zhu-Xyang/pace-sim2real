# S800 分组扫频与分组辨识

本实验在 `s800_single_part` 分支使用现有 S800 资产和增益，参考 G1 实验的指令延时、静止段、淡入淡出和真值回放检查。这里的“分段”指三个身体部位分别采集、分别拟合，不是把一次全身扫频按时间切成三段。

| 组 | 关节 | 关节数 | 优化维数（含一个延时） |
|---|---|---:|---:|
| `legs` | 双腿髋、膝、踝 | 12 | 49 |
| `torso` | `J12_TORSO_YAW` | 1 | 5 |
| `arms` | 双臂肩、肘、手腕，不含手指 | 14 | 57 |

每次只有本组目标扫频，其他关节收到恒定的中心目标。仿真保留全身质量和关节动力学；恒定目标不是机械锁死。CMA-ES 只搜索本组的 armature、粘性阻尼、摩擦、零偏，损失也只计算本组。固定延时时，各组维数再减 1。

## 1. 先明确实验口径

默认分组实验使用 `--delay_mode command`，即先延迟目标位置/速度/前馈力矩，再用当前状态计算 PD 和限幅。`--delay_mode torque` 保留原来的输出力矩延时，可作对照。普通 `--group all` 未指定模式时仍默认 torque。

当前分支的注入延时默认 **3 步 × 2.5 ms = 7.5 ms**，可用 `--delay_steps` 改成 5 步。它是人为注入值，不是 S800 真机实测值。其他 GT 仍为本分支 `data_collection.py` 中的分电机类型参数，没有套用 G1 的统一真值。

默认轨迹为：1 秒恒定目标 → 20 秒 0.1–4 Hz 线性 chirp → 1 秒恒定目标，chirp 首尾各 2 秒淡入淡出；时间轴严格为 `k * dt`。默认使用 `scaled_legacy` 轨迹：恢复原来的归一化 bias、方向和 scale，腿部组 scale 乘 0.8（腰/手臂组默认不缩放）；未激励组保持原中心。详见第 7 节。`--grouped_sweep` 是另一个选项，表示使用逐关节 3/6 Hz 上限，初次实验不建议同时改变这个变量。

`scaled_legacy`/`legacy` 分组默认关闭自碰撞，沿用旧实验设置；不代表轨迹无几何穿透。可选的 `clearance` 开启模型已有的自碰撞，拟合自动继承录制时的设置。`--trajectory_profile legacy` 可复现旧中心/幅值，legacy 分组默认关闭自碰撞。几何检查独立于物理碰撞开关；关闭物理碰撞也不能绕过 clearance 验收。

**torso 和双臂并非独立支链。** 错误的冻结参数可能通过真实运动耦合影响本组。提供两种明确区分的实验：

- 条件恢复实验：`--freeze_unfitted_gt` 仅把未拟合关节的四类参数固定为数据里的真值，本组依然从搜索区间中点或 warm start 开始；延时仍独立搜索，除非使用 `--fix_delay`。用于排除未拟合组参数错误的影响，不能当作未知全参数辨识结果。
- 未知参数实验：不加上述选项，未拟合参数来自 `--warm_start`，没有 warm start 则使用区间中点。可交替细化 torso/arms，再用全身数据做联合验证。不要把只看本组的损失误认为已经消除了物理耦合。

此外，固定基座、仅编码器观测且绕重力轴旋转时，torso yaw 的绝对零偏可能不可观；需要独立几何参考才能确定。不要以这个参数是否恢复到 0.05 来单独判断整个辨识流程成败。

## 2. 一条命令依次跑三组

在项目根目录、已安装 Isaac Lab 的 Python 环境下执行。先跑短程验收：

```bash
conda activate env_isaaclab
python -u scripts/pace/run_s800_groups.py --smoke --freeze_unfitted_gt --fix_delay 3
```

每组顺序执行：采集 1.2 秒 → 8 环境真值回放 → 两代拟合 → 参数报告。任一命令失败或真值回放未达到阈值会停止。短程验收只验证代码路径，不验证参数收敛。

只检查完整 22 秒记录的采集与回放，可运行 `python -u scripts/pace/run_s800_groups.py --check_only --num_envs 8`，不会启动正式优化。

正式条件恢复实验（22 秒记录、每组 100 代；100 代是起始预算，不保证收敛）：

```bash
python -u scripts/pace/run_s800_groups.py \
    --freeze_unfitted_gt --fix_delay 3 --num_envs 256 --max_iterations 100
```

初次固定延时有助于把连续物理参数的恢复与离散延时搜索分开。之后去掉 `--fix_delay 3` 做联合辨识。需要复现旧的 5 步真值时同时设置 `--delay_steps 5 --fix_delay 5`。

每次批量实验使用新的目录，不覆盖之前的正式数据：

```text
data/s800_sim/{smoke|experiment}_<时间戳>/chirp_<组>_<延时模式>.pt
logs/pace/s800_groups/{smoke|experiment}_<时间戳>/
  legs_collect.log / legs_floor.log / legs_fit.log / legs_eval.log
  legs/floor/<时间戳>/
  legs/fit/<时间戳>/
  torso/... 和 arms/...
```

`--groups legs` 可只跑腿；`--groups torso arms` 只跑上身两组。并行环境较多时轨迹缓冲区消耗较大，先从 256 开始，根据可用显存调整。

## 3. 分步运行与调参

以下为腿部，换成 torso 或 arms 即可。单独运行采集器遇到同名数据时，会把旧 `.pt` 和对应轨迹 JSON 改名为 `.previous_<时间戳>` 后再保存。频率/增益对照仍建议使用 `--output` 指定新路径。

```bash
python -u scripts/pace/data_collection.py --headless --group legs \
    --delay_mode command --delay_steps 3 --duration 20 --max_frequency 4

python -u scripts/pace/fit.py --headless --group legs --num_envs 32 --floor_test

python -u scripts/pace/fit.py --headless --group legs --num_envs 256 \
    --freeze_unfitted_gt --fix_delay 3 --max_iterations 100 --save_interval 10
```

--fix_delay 需要根据实际采集的情况来制定，加上此参数说明delay不参与优化固定在制定步树，去掉就会参与优化

默认数据为 `data/s800_sim/chirp_legs_command.pt`，日志为 `logs/pace/s800_sim_legs_command/<时间戳>/`。拟合打印的 `[group] active=...` 应与上表对应。

直接运行 `data_collection.py` 时，采集完成后默认显示原有的三张诊断图：实际位置与目标位置、施加力矩与扣除摩擦后的净力矩、关节速度。图中保留全部关节，方便检查未激励组的运动。需要无人值守时加 `--no-plot`；`--headless` 只关闭仿真窗口，不关闭这三张图。一键批量脚本 `run_s800_groups.py` 会显式传入 `--no-plot`。

指定一份数据并跟随录制的执行器模式：

```bash
python -u scripts/pace/fit.py --headless --group arms \
    --data s800_sim/experiment_<时间戳>/chirp_arms_command.pt \
    --freeze_unfitted_gt --fix_delay 3 --num_envs 256 --max_iterations 100
```

不使用冻结真值时，可用前一轮结果初始化下一轮，例如：

```bash
python -u scripts/pace/fit.py --headless --group torso \
    --warm_start logs/pace/<上一轮目录>/best_params.pt \
    --fix_delay 3 --num_envs 256 --max_iterations 100
```

每份参数文件仍保存完整 109 维，但只有本轮选中的维度经过辨识。三组独立的条件恢复实验不能直接把某一组文件当作全身辨识结果；其中其他组只是已知真值或先验。

## 4. 验收与报告

```bash
python scripts/pace/eval_group.py --run logs/pace/s800_sim_legs_command/<拟合时间戳>
```

报告只显示本组参数，GT 取自该次数据快照，不使用手写对照表；另存 `group_comparison.csv`。延时同时报告原始连续值与实际整数步数，避免把 `3.2 → 3` 的离散结果误报为延时误差。

默认评价 `best_params.pt`，它与 `best_trajectory.pt` 对应，均为最后一代的最优已评估个体；不是所有历史代的全局最优。`mean_*.pt` 是 CMA-ES 分布均值，尚未单独回放，不能把最优个体轨迹当成它的轨迹。可用 `--params mean_090.pt` 单独检查均值参数。

拟合开始前检查：关节顺序、所选关节是否激励、数据采样间隔、延时结构、增益及执行器限值。数据记录 `joint_names`、`excited`、`experiment` 和 `gt`；`config.pt` 还记录本轮拟合组、固定延时、冻结参数来源等。新数据有静止段和淡入淡出，不再套用旧的“整段时间线性映射频率”标签。

正式验收应先通过整个记录的真值回放，再检查参数恢复、残差、不同初始化以及新轨迹的预测。短时间开局吻合不足以证明全程一致。力矩延时对照需要用 torque 模式重新采集和拟合，不能复用 command 模式数据来比较同结构恢复效果。

CPU 回归检查（不启动仿真器）：

```bash
python -m unittest discover -s scripts/pace -p 'test_*.py'
```

## 5. 本分支验证记录（2026-10-09）

以下为增加间隙检查之前的 legacy 轨迹验证记录。6 项 CPU 回归检查通过，覆盖真实 S800 的 12/1/14 分组、恒定目标与时间轴、冻结真值不泄漏到本组初始化、录制配置校验、损失掩码与参数/轨迹对应，以及指令延时使用当前反馈的语义。

三组各完成短程采集、8 环境真值回放、两代固定延时的条件恢复拟合和报告导出。记录在 `logs/pace/s800_groups/smoke_26_10_09_15-45-39/`，两代结果仅用于检查流程。

完整 22 秒、0.1–4 Hz、command 延时 3 步的数据已保存到 `data/s800_sim/experiment_26_10_09_15-49-25/`。每组 8,800 帧、27 列；已核对未激励组的指令全程恒定，位置记录均为有限值。8 环境真值回放结果：

| 组 | 最小损失 | 最大损失 | 阈值检查 |
|---|---:|---:|---|
| legs | 1.635e-13 | 3.384e-13 | 8/8 通过 |
| torso | 2.680e-15 | 5.334e-15 | 8/8 通过 |
| arms | 9.677e-14 | 1.407e-13 | 8/8 通过 |

损失是该组关节位置平方误差和的时间平均，单位 rad²。日志在 `logs/pace/s800_groups/experiment_26_10_09_15-49-25/`。这验证了完整记录的回放一致性，尚未验证正式拟合收敛。

以下旧数据仅用于复现无接触约束加入前的参数恢复基线，不是无接触轨迹验收结果。复现命令：

```bash
for part in legs torso arms; do
    python -u scripts/pace/fit.py --headless --group "$part" \
        --data "s800_sim/experiment_26_10_09_15-49-25/chirp_${part}_command.pt" \
        --freeze_unfitted_gt --fix_delay 3 --num_envs 256 \
        --max_iterations 100 --save_interval 10 --epsilon 0 || break
done
```


## 6. 腿部间隙约束轨迹（可选，已取消默认）

几何检查使用 `trimesh` 读取 STL，已加入项目安装依赖；旧环境若缺少该包，运行 `python -m pip install -e source/pace_sim2real` 更新。

这一外展方案已不再默认使用，因为 ±0.6 rad 的髋侧摆中心带来了明显的静态跟踪偏差。以下仅用于显式复现该方案，仍显示三张图：

```bash
python -u scripts/pace/data_collection.py --group legs --delay_mode command \
    --delay_steps 3 --duration 20 --max_frequency 8 --trajectory_profile clearance
```

`q_cmd = center_rad + signed_amplitude_rad * chirp`。中心角度与幅值均为弧度，编码器零偏仍是独立的 GT 参数，不要与原先的归一化 trajectory_bias 混淆。armature、阻尼、摩擦、编码器零偏和控制增益没有因本次轨迹设计而改变。

默认规划从左右髋侧摆中心 `+0.6/-0.6 rad` 起筛选，必要时增至 `+0.65/-0.65 rad`。腿部基础幅值降至原值 60%，髋偏航、踝侧摆进一步限制：

| 单侧关节 | 原幅值绝对值 rad | 新默认候选最大幅值 rad |
|---|---:|---:|
| 髋俯仰 | 0.5 | 0.3 |
| 髋侧摆 | 0.3 | 0.18 |
| 髋偏航 | 0.7 | 0.21 |
| 膝俯仰 | 0.5 | 0.3 |
| 踝俯仰 | 0.272 | 0.1632 |
| 踝侧摆 | 0.105 | 0.0315 |

左右指令保持原有镜像符号。规划器可以进一步减小髋侧摆/偏航幅值。最终中心和幅值以终端打印及每份数据旁的 `.trajectory.json` 为准。减小幅值可能影响参数可辨识性，仍需检查拟合结果和独立轨迹预测；通过间隙检查不等于保证拟合精度。

检查分为两步：

1. **采集前**：URDF 正运动学计算目标位置加 GT 零偏对应的物理姿态。候选先稀疏筛选，再对全部采样和相邻采样的关节角中点检查。目标要求默认至少 **5 cm** 保守间隙（2 cm 验收间隙 + 3 cm 跟踪预留）。
2. **采集后**：对实际记录位置加 GT 零偏、以及最后一步状态重复检查；要求至少 **2 cm** 间隙、**0.01 rad** 关节限位余量。目标轨迹通过，实际轨迹仍可能因过冲而失败；失败会保存为 `.rejected.pt`，拟合脚本会拒绝该记录，不替换此前通过的数据。

`--clearance_margin 0.02` 设置实际间隙阈值。改变频率、延迟、GT 或增益后都需重新验收。可编辑保存的 JSON 的 `center_rad` / `signed_amplitude_rad`，通过 `--trajectory_plan <路径>.trajectory.json` 重用；关节顺序必须一致，并且仍执行两步检查。`--trajectory_profile legacy` 不执行这些检查，只用于复现旧实验。

几何检查覆盖左右大腿、小腿、踝俯仰外壳与足部的 **16 对跨腿组合**、这些部件离地间隙和腿部关节限位。每个包围盒同时包住 URDF 显示网格与已有碰撞体，补足踝俯仰外壳没有独立碰撞体的问题。正数是几何距离的保守下界；负数只表示包围盒重叠，不能直接解释为真实网格穿透深度。

这仍是固定基座条件下的离散几何检查，未覆盖同侧相邻部件、手臂与躯干接触，也不是连续时间或真机悬吊摆动的无碰撞证明。中点加密只是额外采样。`torso`/`arms` 采集也检查静止腿部，但不能因此声称上半身轨迹已通过全身无接触验证。真机编码器零偏未知，不能直接照搬这里使用已知 GT 的姿态校正。


### 新轨迹的实际回放记录（2026-10-09）

20 秒扫频 + 首尾各 1 秒静止，0.1–8 Hz、command 延时 3 步、自碰撞开启。目标中心髋侧摆为 ±0.6 rad，幅值为上表新默认最大值，其余中心沿用原值。记录有 8,800 帧，实际检查还包含末步状态和相邻帧中点：

- 跨腿几何距离保守下界最小值：**0.064815 m**，发生在约 **4.77125 s** 的左右大腿包围盒之间。
- 被检查部件最小离地间隙：**0.466877 m**。
- 腿部最小关节限位余量：**0.125399 rad**。
- 数据：`data/s800_sim/experiment_26_10_09_16-47-43/chirp_legs_command.pt`；轨迹参数及检查报告在同名 `.trajectory.json`。
- 8 环境真值回放：全程损失 min **2.570e-13**、max **7.354e-12 rad²**，8/8 低于 `1e-10` 阈值。8 项 CPU 回归和 13 对关节的名义指令镜像检查通过。

原来的 `data/s800_sim/chirp_legs_command.pt` 没有在本次验证中替换。前几轮未通过的候选保存在独立实验目录的 `.rejected.pt`，不能用于拟合。当前通过的是上述具体配置与检查范围；正式参数恢复优化尚需单独运行。


直接拟合这份已通过检查的 8 Hz 新数据（无需再次采集）：

```bash
python -u scripts/pace/fit.py --headless --group legs --num_envs 256 \
    --data s800_sim/experiment_26_10_09_16-47-43/chirp_legs_command.pt \
    --freeze_unfitted_gt --fix_delay 3 --max_iterations 100 --save_interval 10
```


默认 0.1–4 Hz 的完整三组兼容性检查使用独立目录 `data/s800_sim/experiment_26_10_09_16-49-35/`（日志同名目录在 `logs/pace/s800_groups/`）。腿部实际保守间隙 **5.87 cm**；其真值回放损失 min **1.733e-13**、max **1.078e-10 rad²**，**7/8** 环境严格低于 `1e-10`。一个环境略超阈值，不能描述为 8/8 通过；尚未定位该数值差异的来源。Torso 组腿部保持姿态检查通过，真值回放 min **3.280e-15**、max **4.590e-15 rad²**，8/8 通过。

三张原有诊断图还使用已通过的 8 Hz 数据在非交互绘图后端执行过检查，确认各有 27 个关节、分别绘制两条位置曲线、两条力矩曲线和一条速度曲线。

Arms 组也完成完整 22 秒采集与真值回放：腿部保持姿态的最小保守间隙 **8.67 cm**，真值回放 min **7.510e-13**、max **1.903e-11 rad²**，8/8 通过。此处没有检查手臂与躯干之间的间隙。


## 7. 恢复原轨迹，仅降低 scale（当前默认）

最新用户数据 `chirp_legs_command.pt`（2026-10-09 16:56）实际上使用 0.1–4 Hz。外展方案的 GT 编码器 bias 仍为 0.05，但最后 0.5 秒静止段 `q_encoder - q_target` 的均值在 J01/J02/J07 上分别达到 -0.1327/+0.2047/+0.1182 rad。这是位置跟踪偏差，不是已经辨识出的编码器 bias；较大的姿态负载和现有 PD 控制会产生静态误差，不能仅凭曲线偏离量反推编码器零偏。

已将默认方案改成 `scaled_legacy`，不再自动把髋侧摆中心外移。严格恢复旧公式：

```text
q_cmd = (chirp + original_trajectory_bias) * original_direction * (original_scale * scale_factor)
```

腿部组默认 `scale_factor=0.8`。因此实际中心和幅值同时乘 0.8，原始归一化 trajectory_bias 保持不变，编码器 GT bias 完全不变。J01/J07 中心现在是 ±0.036 rad；膝中心 0.4184 rad。未激励关节保持原中心。上身组默认 factor=1；显式指定 `--scale_factor` 可缩放当前组。

```bash
python -u scripts/pace/data_collection.py --group legs --delay_mode command \
    --delay_steps 3 --duration 20 --max_frequency 4 --scale_factor 0.8
```

三张图仍默认显示。同名旧数据仍自动备份。`--trajectory_profile legacy` 精确恢复未缩放原轨迹；`--trajectory_profile clearance` 复现上一版大外展方案。`--scale_factor` 不允许与这两种 profile 或 `--trajectory_plan` 混用，避免参数被静默忽略。

原轨迹及缩放方案没有无接触保证，不执行外展方案的间隙验收，也不改变原有的物理碰撞开关。离线几何筛选已经确认，单纯把 scale 降低 20% 不足以保证腿间无穿透。它用于恢复原姿态下的参数辨识对照；若实验要求严格无接触，需要继续设计运动组合，不能仅凭这次缩放认为要求已满足。


### 80% scale 实测对照

完整 22 秒、0.1–4 Hz、command 延时 3 步；GT、增益不变，物理自碰撞沿用旧版关闭设置。最后 0.5 秒静止段的平均 `编码器位置 - 目标位置`（rad）：

| 关节 | 大外展方案 | 原轨迹 scale × 0.8 |
|---|---:|---:|
| J01 | -0.13273 | -0.01817 |
| J02 | +0.20473 | +0.02366 |
| J05 | -0.01073 | -0.00098 |
| J07 | +0.11817 | +0.00155 |
| J08 | -0.17596 | +0.00846 |
| J11 | +0.00713 | -0.00355 |

这些数值是跟踪误差，不是 PACE 辨识结果。大外展记录中 J01 的静止力矩约 +26.55 Nm，与 `-kp*error = -200*(-0.13273)` 一致；减小 scale 并恢复原姿态后约 +3.63 Nm。数据支持明显的 PD 静态负载误差，不能把它直接视为错误的编码器 bias。外展和原版的自碰撞开关不同，本对照不是只改变一个变量的严格因果实验。

大外展方案的踝侧摆幅值仅 0.0315 rad，摩擦/刚度的等效误差尺度约 `0.3/17.16 = 0.0175 rad`，小幅激励易出现摩擦相关平段。回退后的幅值为 0.084 rad。动态跟踪并未因此完美：J02 的全程 RMS 误差仍为 0.4058 rad（未缩放旧轨迹为 0.4929 rad）；扫频过冲仍需与静态偏差分别看待。

新数据保存在 `data/s800_sim/scale_comparison/legs_scale080_4hz.pt`，没有覆盖用户的最新大外展记录。对照图与逐关节统计分别为 `logs/pace/s800_scale_comparison/tracking_comparison.png`、`tracking_comparison.csv`。图的三列依次为大外展、原轨迹、原轨迹 ×0.8。

实际几何诊断出现跨腿包围盒重叠，最小腿部限位余量约零（触及限位），所以这里只完成了恢复原轨迹并改善静态偏差的对照，未实现无接触/远离限位。负的包围盒间隙不等于真实网格穿透深度。

复用该数据拟合时显式指定文件，避免误用根目录中旧的大外展数据：

```bash
python -u scripts/pace/fit.py --headless --group legs --num_envs 256 \
    --data s800_sim/scale_comparison/legs_scale080_4hz.pt \
    --freeze_unfitted_gt --fix_delay 3 --max_iterations 100 --save_interval 10
```

新缩放数据的 8 环境完整真值回放通过：损失 min **1.011e-13**、max **1.796e-13 rad²**，8/8 低于 `1e-10`。9 项 CPU 回归检查通过。尚未运行正式 100 代辨识。


## 8. 手臂数据诊断（2026-10-10）

检查文件：`data/s800_sim/chirp_arms_command.pt`，0.1–4 Hz、22 秒、14 个手臂关节同时激励，command 延迟 **3 步**，scale factor=1，自碰撞关闭。它与 `logs/pace/s800_sim_arms_command/26_10_09_18-34-32/config.pt` 中的目标位置和观测位置逐点完全相同，虽然当前文件修改时间更新。

### 已确认的拟合配置问题

上述拟合运行把延迟固定为 **5 步**，与该次数据快照中的 GT=3 不一致。运行保存到了 `mean_999.pt`；其 `best_params.pt` 的左右肘俯仰 armature 为约 **0.0050006 / 0.0050003**，接近搜索下界 0.005，GT 则为 0.05。不能据此认为手臂真实惯量小了 90%，或单纯归因于优化代数不够。完整最优轨迹的手臂位置平方误差和时间均值约 **0.037884 rad²**。

原 `plot_trajectory.py` 手写 `GT_DELAY=5` 会把这一运行的 Delay 行错误显示为真值 5。现已修正为优先读取该运行 `config.pt` 的 GT（包含各关节 bias），只对没有 GT 的旧记录使用手写兜底值并警告。`fit.py` 也在初始化环境前打印数据路径、数据 GT 延迟和本轮延迟设置，固定值不一致时警告；不会静默改写用户指定值。

### 轨迹问题与拟合问题分开看

| 关节 | 目标峰峰值 rad | 实际峰峰值 rad | 达到 99% 固定力矩上限的采样比例 |
|---|---:|---:|---:|
| J18 wrist_pitch L | 0.100 | 0.872 | 4.95% |
| J32 wrist_pitch R | 0.100 | 0.872 | 10.28% |
| J19 wrist_roll L | 0.100 | 1.209 | 0% |
| J33 wrist_roll R | 0.100 | 1.194 | 0% |

四个腕关节都有触及 URDF 限位的样本。wrist_roll 的固定力矩上限未饱和，不应把所有腕部问题都归因于力矩饱和；其大幅运动提示可能存在全臂同时激励造成的动力学耦合/振动放大，各因素的贡献尚需隔离实验确认。两侧 elbow_pitch / elbow_yaw 均没有贴近限位或达到固定力矩上限，不能与腕部一概而论。上述上限统计不等同于所有可能的速度相关限幅事件。

### 同数据真值回放

8 环境全程真值回放通过：损失 min **1.034e-13**、max **2.906e-13 rad²**，8/8 低于 `1e-10`。日志：`logs/pace/arms_diagnostic_20261010/floor.log`。这说明当前模型能够复现记录里的实际运动，不证明 56 个未知动力学参数都能达到 1% 恢复误差。

图表：`logs/pace/arms_diagnostic_20261010/elbow_wrist_diagnostic.png`；逐关节统计：同目录 `trajectory_metrics.csv`。关节限位判断使用物理位置 = 编码器位置 + GT bias，图中的限位已转换到编码器坐标系。

优先用原数据重跑正确延迟，保留其他条件，避免同时改轨迹和增益：

```bash
python -u scripts/pace/fit.py --headless --group arms --num_envs 256 \
    --data s800_sim/chirp_arms_command.pt \
    --freeze_unfitted_gt --fix_delay 3 \
    --max_iterations 100 --save_interval 10 --epsilon 0
```

这里的 3 来自这份手臂数据，不能套用腿部数据的 5。100 代只是起始预算，尚未运行这轮更正后的正式拟合。若更正延迟后仍有恢复困难，下一项应做保持肩部/近端目标不扫频、单独激励肘腕的对照，以检查近端激励导致的腕部放大，再决定是否调整各关节幅值/带宽；不是直接提高所有增益或放宽力矩限值。
