# PACE 公式与代码实现分析

本文档分析 PACE 论文公式 (6) 与 `pace-sim2real` 代码实现的对应关系，以及 S800 参数审计中发现的关键问题。

---

## 1. 论文公式 (6) 与代码映射

论文公式 (6)：

```
Ia·q̈ + d·q̇ = sat(Pτ·(q̂ - q + q̃_bias) - Dτ·q̇ + τ_comp) + τ_f
```

代码分两层实现：

### 第 1 层 — `PaceDCMotor.compute()`（Python 层，显式力矩计算）

| 公式项 | 代码位置 | 对应 |
|--------|---------|------|
| `Pτ·(q̂ - q + q̃_bias)` | `IdealPDActuator`: `kp·(q_des - (q - encoder_bias))` | ✅ 完全对应 |
| `-Dτ·q̇` | `IdealPDActuator`: `kd·(q̇_des - q̇)`，因为 chirp 只发位置指令，`q̇_des=0` | ✅ 完全对应 |
| `τ_comp` | `control_action.joint_efforts`，但 data_collection 的 chirp 不发 feedforward | ⚠️ `τ_comp = 0` |
| `sat()` | `DCMotor._clip_effort`：DC 电机扭矩-速度饱和曲线 | ✅ 完全对应 |
| delay | `DelayBuffer`（公式未显式写出） | ➕ 额外建模 |

### 第 2 层 — PhysX 物理引擎（隐式动力学积分）

| 公式项 | 代码 | 对应 |
|--------|------|------|
| `Ia·q̈` | `write_joint_armature_to_sim()` → PhysX joint armature | ✅ 完全对应 |
| `d·q̇` | `write_joint_viscous_friction_coefficient_to_sim()` → PhysX viscous friction | ✅ 完全对应 |
| `τ_f` | `write_joint_friction_coefficient_to_sim()` (static) + `write_joint_dynamic_friction_coefficient_to_sim()` (dynamic) | ⚠️ 见下文 |

### 数据流总结

```
data_collection.py
  │
  ├── chirp 位置指令 q̂ (正弦扫频信号)
  │
  ▼
PaceDCMotor.compute()
  ├── 1. encoder bias:  q_encoder = q - encoder_bias       → q̃_bias
  ├── 2. PD 力矩:       τ = kp·(q̂ - q_encoder) - kd·q̇      → Pτ·(q̂-q+q̃_bias) - Dτ·q̇
  ├── 3. DC 饱和:        τ_clipped = clip(τ, τ_min(q̇), τ_max(q̇))  → sat()
  └── 4. 延迟:           τ_delayed = DelayBuffer(τ_clipped, delay)  → 额外建模
  │
  ▼ τ_delayed 写入 PhysX joint effort
PhysX 物理引擎
  ├── armature:          (I_link + I_armature)·q̈           → Ia·q̈
  ├── viscous friction:  d_viscous·q̇                       → d·q̇
  └── coulomb friction:   τ_friction·sign(q̇) (stiction model) → τ_f
```

---

## 2. 关键差异：τ_f 的实现

论文公式写的是 `+τ_f`，暗示库仑摩擦 `τ_f·sign(q̇)` 是简单的加法项，假设关节始终在运动。

代码中 PhysX 的实现是 **stiction + sliding** 模型：

```
PhysX 实际行为:
  if |τ_applied| < τ_static:  关节不动 (stiction 死区)
  else:                        τ_net = τ_applied - τ_dynamic·sign(q̇)
```

`data_collection.py` 设 `static = dynamic = friction`，所以是纯库仑摩擦。但这与公式的 `+τ_f` 有本质区别：

- **公式假设**关节一直在动（`sign(q̇)` 始终有值），摩擦只是恒定阻力
- **PhysX 实现**在力矩小于 static friction 时会**完全卡住关节**，产生死区

### 对小关节的影响

当 friction 值相对于 chirp 力矩较大时，stiction 死区会导致信号失真：

| 关节 | kp | chirp scale | chirp 最大力矩 | friction | friction 占比 | 状态 |
|------|-----|------------|--------------|----------|-------------|------|
| HIP_PITCH | 240 | 0.500 | 120.0 Nm | 1.0 | 0.8% | ✅ 无影响 |
| KNEE_PITCH | 240 | 0.500 | 120.0 Nm | 1.0 | 0.8% | ✅ 无影响 |
| SHOULDER_PITCH | 90 | 0.400 | 36.0 Nm | 1.0 | 2.8% | ✅ 无影响 |
| ELBOW_YAW | 50 | 0.350 | 17.5 Nm | 1.0 | 5.7% | ✅ 无影响 |
| ANKLE_ROLL | 90 | 0.105 | 9.5 Nm | 1.0 | 10.6% | ✅ 可接受 |
| **WRIST_PITCH** | **12.5** | **0.150** | **1.88 Nm** | **1.0** | **53.3%** | ❌ 严重失真 |
| **WRIST_ROLL** | **12.5** | **0.150** | **1.88 Nm** | **1.0** | **53.3%** | ❌ 严重失真 |

手腕关节 chirp 最大力矩仅 1.88 Nm，而 friction=1.0 Nm 占了 53%。chirp 正弦信号过零时 `|τ| < 1.0`，关节卡住 → stick-slip → 信号非线性失真 → armature 和 friction 参数耦合，辨识退化。

---

## 3. S800 参数审计结果

### 3.1 参数值全部正确（无 Anymal 残留）

| 检查项 | 文件 | 结果 |
|--------|------|------|
| Armature (27 关节) | `data_collection.py` vs `kpkd.py` | ✅ 全部匹配 |
| Stiffness/kp (14 类型) | `s800_pace_env_cfg.py` vs `kpkd.py` | ✅ 全部匹配 |
| Damping/kd (14 类型) | `s800_pace_env_cfg.py` vs `kpkd.py` | ✅ 全部匹配 |
| Effort_limit (10 类型) | `s800_pace_env_cfg.py` vs `kpkd.py` | ✅ 全部匹配 |
| Velocity_limit (10 类型) | `s800_pace_env_cfg.py` vs `kpkd.py` | ✅ 全部匹配 |
| Friction/viscous/dynamic | actuator cfg = 1.0/1.6/1.0 | ⚠️ 值合理但来源存疑（见 3.3） |
| CMA-ES bounds vs GT | 所有 GT 值在 bounds 内，居中 20%-75% | ✅ 合理 |
| GT_ARMATURE (plot_trajectory) | 27 关节全部存在且匹配 | ✅ 正确 |
| Joint order | J00-J12 腿+腰, J13-J19 左臂, J27-J33 右臂 | ✅ 正确 |
| Task 注册 | `Isaac-Pace-S800-v0` 已在 `__init__.py` 注册 | ✅ 正确 |
| Chirp 方向/偏置/幅度 | 27 关节 S800 专用值 | ✅ 正确 |
| USD 路径 & fix_root_link | 正确 S800 路径, fix_root_link=False | ✅ 正确 |

### 3.2 发现的 Anymal 残留 Bug

| 优先级 | 问题 | 位置 | 说明 |
|--------|------|------|------|
| **P0** | `fit.py` 默认 `--task=Isaac-Pace-Anymal-D-v0` | `scripts/pace/fit.py` | 应改为 `Isaac-Pace-S800-v0` |
| **P0** | `plot_trajectory.py` 默认 `--robot_name=anymal_d_sim` | `scripts/pace/plot_trajectory.py` | 应改为 `s800_sim` |
| P3 | 过时注释 `friction between 0.0 - 0.5` | `s800_pace_env_cfg.py:139` | 实际 bounds 为 [0.1, 2.0] |
| P3 | 过时注释 `dof_damping between 0.0 - 7.0` | `s800_pace_env_cfg.py:136` | 实际 bounds 为 [0.3, 4.0] |

### 3.3 friction/damping 值来源存疑

MuJoCo XML (`serial_robot_s.xml`) 中有三个不同的 "friction" 概念：

| 来源 | 参数 | 值 | 含义 |
|------|------|-----|------|
| `<joint damping="0.6">` | damping | 0.6 | 关节粘性阻尼 (Nm·s/rad) → 对应 IsaacLab `viscous_friction` |
| `<joint frictionloss="0.8">` | frictionloss | 0.8 | 关节库仑摩擦 (Nm) → 对应 IsaacLab `friction`/`dynamic_friction` |
| `<geom friction="1.0 0.005 0.0001">` | friction | 1.0 | **接触**摩擦（无量纲）→ 与关节摩擦**无关** |

`s800_pace_env_cfg.py` 中的映射：

```python
friction={".*": 1.0},          # 注释: "from XML"  ← 取了 <geom friction="1.0">，是接触摩擦，不是关节摩擦
viscous_friction={".*": 1.6},  # 注释: "from XML"  ← XML 中关节 damping=0.6，1.6 来源不明
```

**正确映射应为**：`friction=0.8`（来自 `frictionloss`），`viscous_friction=0.6`（来自 `damping`）。

与 Anymal 对比：

| 参数 | Anymal | S800 (当前) | S800 (XML 真实值) | 单位 |
|------|--------|------------|-------------------|------|
| 关节库仑摩擦 | 0.05 | 1.0 ❌ | 0.8 | Nm |
| 关节粘性阻尼 | 4.5 | 1.6 ❌ | 0.6 | Nm·s/rad |

**注意**：XML 中所有关节的 friction/damping 统一，这与实际硬件不符（不同关节电机规格不同）。参考 G1、H1 等人形机器人 XML，不同关节的 friction 值均不同且小于 0.5。

---

## 4. PhysX friction 写入顺序修复

**问题**：`data_collection.py` 设置 friction 时顺序错误，导致 PhysX 报错 "Static friction effort must be greater than or equal to dynamic friction effort"。

**根因**：actuator 配置默认 `friction=1.0, dynamic_friction=1.0`，`data_collection.py` 先写 static=0.0 时 dynamic 仍=1.0，违反 `static >= dynamic` 约束。

**修复**：交换写入顺序，先写 dynamic 再写 static（与 `cma_es.py` 中已有的 workaround 一致）：

```python
# 修复前（报错）:
write_joint_friction_coefficient_to_sim(0.0)           # static=0, dynamic 仍=1.0 → 0 < 1.0 ❌
write_joint_dynamic_friction_coefficient_to_sim(0.0)    # dynamic=0

# 修复后（安全）:
write_joint_dynamic_friction_coefficient_to_sim(0.0)    # dynamic=0, static 仍=1.0 → 1.0 >= 0 ✅
write_joint_friction_coefficient_to_sim(0.0)            # static=0, dynamic=0 → 0 >= 0 ✅
```

此修复不影响最终参数值，只改变写入顺序的中间状态。

---

## 5. 优化质量建议

| 优先级 | 问题 | 建议 |
|--------|------|------|
| P1 | `fit.py` 默认 `--num_envs=4096` | 降至 64-128（109 参数推荐 population ≈ 4+3·ln(109) ≈ 18） |
| P2 | score 函数为无权重 MSE | 按 chirp 幅度归一化，避免大关节主导小关节 |
| P2 | 所有关节 friction/damping 统一值 | 改为 per-joint 值（参考 kpkd.py 的 per-joint armature 模式） |
| P3 | friction bounds 下界 0.1 | 如果真机小关节 friction < 0.1，CMA-ES 无法搜索到真值 |

### 关于 L-R 对称约束

**不建议**强制左右对称。sim2real 辨识的核心目的是找真实硬件与设计值之间的 gap，真实机器人左右侧确实会因制造公差、装配差异、磨损等因素存在差异。如果强制 L=R，优化器会输出一个左右平均的参数集，两边都不准确。当前 109 个独立参数的设计是正确的。
