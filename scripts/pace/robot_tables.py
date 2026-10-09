"""逐机器人的「注入真值 + 激励设计」——**单一数据源**。

想改一个参数，只看这张表
------------------------
每个机器人一张表，含义固定：

    armature      注入仿真的电枢惯量真值 [kg m^2]
    damping       viscous friction 真值 [Nm s/rad]
    friction      Coulomb friction 真值 [Nm]
    f1_per_joint  分组扫频上限 [Hz]（--grouped-sweep 关闭时被全局 f1 覆盖）
    directions    激励方向 ±1（镜像对称，见下）
    bias          激励偏置 [rad]（把 sin 抬到行程中间）
    scale         激励幅度 [rad] = min(半行程×0.4, 力矩权限, 物理上限)
    bias_gt       编码器零位偏差真值 [rad]（全场统一）
    delay_gt      指令延时真值 [仿真步]（全场统一）
    excite        分组激励（--excite legs/upper）时的幅度与扫频上限覆盖，见该键的注释

**数组顺序 = JOINT_ORDER[robot]**（= env cfg 的 joint_order）。每行末尾都标了
对应的关节名，按肢体分块；不要靠数下标定位，数错过。
命令行看「逐关节展开」的版本（推荐）：

    python scripts/pace/robot_tables.py --robot g1

被四处共用，避免各存一份而漂移：
  - scripts/pace/data_collection.py   （注入真值时）
  - scripts/pace/plot_trajectory.py   （出参数对比表时，run 的 config.pt 没带 gt 的老 run 回落用）
  - scripts/check_mirror_symmetry.py  （校验 directions/bias/scale 的镜像对称性）
  - scripts/pace/fit.py               （关节分组规则 GROUP_RULES / joint_group()）
本模块**不能** import isaaclab / torch —— data_collection 里的那段会启动 Isaac Sim，
所以这里是纯字面量，谁都能 import。

⚠️ 改这里的值 = 改注入仿真的真值。改完要重采数据，否则数据里的 gt 和新表对不上
   （chirp_data.pt 会记下注入时的值，plot_trajectory 优先用那份记录，所以不会算错，
    只是两者不一致时以数据里的为准）。
⚠️ armature/damping/friction 的真值必须落在各机器人 env cfg 的 `bounds_params` 内，
   越界时 CMA-ES 会贴着边界收敛**且不报错**。data_collection.py 启动时会自动查
   （贴边 <5% 也会告警）。
"""

import re

# ── 关节分组（腿 / 腰 / 臂）────────────────────────────────────────────────
# 大小写无关，顺序即优先级，先匹配到的先归组。S800 的 "J00_HIP_PITCH_L" 与
# G1 的 "left_hip_pitch_joint" 共用同一套规则（都是子串匹配）。
#   legs  : 髋 / 膝 / 踝
#   torso : S800 只有 TORSO_YAW；G1 有 waist_{yaw,roll,pitch} 三个
#   arms  : 肩 / 肘 / 腕
# 分组的物理依据：基座焊死 ⇒ 质量阵在支链（左腿 / 右腿 / [腰→双臂]）间块对角，
# 所以腿和臂可以分开辨识；但**腰和臂必须同一轮**（腰是臂的父关节），
# 对应 `--group arms,torso`，也可写成 "upper"。
GROUP_RULES = (
    ("legs", ("hip", "knee", "ankle")),
    ("torso", ("torso", "waist")),
    ("arms", ("shoulder", "elbow", "wrist")),
)
GROUP_ALIASES = {"upper": ("torso", "arms")}   # G1 sim2sim 手册用的叫法


def joint_group(name: str) -> str:
    """关节名 → 'legs' | 'torso' | 'arms'。落不进任何一组直接报错（不静默漏掉）。"""
    low = name.lower()
    for gname, keys in GROUP_RULES:
        if any(k in low for k in keys):
            return gname
    raise ValueError(f"关节 {name} 落不进任何一组，请更新 robot_tables.GROUP_RULES")


def joint_type(name: str) -> str:
    """'left_hip_pitch_joint' -> 'HIP_PITCH'；'J00_HIP_PITCH_L' -> 'HIP_PITCH'。

    与 g1_pace_env_cfg.g1_joint_type 同一套口径（那边只处理 G1 的拼法）。
    """
    m = re.match(r"^J\d+_(.+?)(?:_[LR])?$", name)
    if m:
        return m.group(1)
    s = name
    for pre in ("left_", "right_"):
        if s.startswith(pre):
            s = s[len(pre):]
    if s.endswith("_joint"):
        s = s[: -len("_joint")]
    return s.upper()


def group_indices(joint_order: list[str], *groups: str) -> list[int]:
    """任意 joint_order 列表里属于这些组的关节下标。支持 "upper" 别名、"all"。"""
    if "all" in groups:
        return list(range(len(joint_order)))
    expanded = []
    for g in groups:
        expanded.extend(GROUP_ALIASES.get(g, (g,)))
    want = set(expanded)
    return [i for i, n in enumerate(joint_order) if joint_group(n) in want]


def joints_in_group(robot: str, *groups: str) -> list[int]:
    """该机器人属于这些组的关节下标（按 JOINT_ORDER 顺序）。"""
    return group_indices(JOINT_ORDER[robot], *groups)


# 每张表的键顺序（三个消费者都按这个顺序读；写表时也照这个顺序排，便于对照）
TABLE_KEYS = (
    "armature", "damping", "friction",          # 注入真值
    "f1_per_joint", "directions", "bias", "scale",  # 激励设计
    "bias_gt", "delay_gt",                      # 全场统一的两项
    "excite",                                   # 可选：分组激励的幅度/频带覆盖
)
# 前 7 个是「逐关节数组」，长度必须等于关节数；后面几个是标量或字典
PER_JOINT_KEYS = TABLE_KEYS[:7]

# directions 的判据（check_mirror_symmetry.py 执行的就是这条）
#   矢状面反射 M = diag(1,-1,1)，a = 关节在 q=0 时的**世界系**转轴：
#       a_R == +M·a_L  →  dir_R = -1   （轴落在镜面内：roll / yaw）
#       a_R == -M·a_L  →  dir_R = +1   （轴沿镜面法向：pure pitch）
#   ⚠️ 不要用「两个轴向量是否相同」当判据 —— 那只在轴落在镜面内时成立。纯 pitch 轴
#      如 (0,1,0) 满足 M·a_L = -a_L，两向量「看起来一样」，但它恰恰是需要 +1 的镜像情形。
#   ⚠️ 每台机器人的转轴不同，方向数组**不能照抄**（G1 的肩偏航/腕部与 S800 相反，
#      且 G1 的髋轴是斜的 [0.98,0,0.17]，不能按名字猜）。
#   bias / scale 左右必须相等：镜像由 directions 承载，靠 bias 反号去凑会得到「反相」而非镜像。

ROBOT_TABLES = {
    # ══ S800（27 关节）══════════════════════════════════════════════════════
    "s800_sim": {
        # ── 注入真值：armature / damping / friction ────────────────────────
        # 这一组是「注入仿真的已知真值」，用来验证 fit 能否恢复。不追求物理正确，要求：
        #   1) 分组内 uniform、组间拉开 → 既能辨识，又能暴露关节间串扰
        #      （27 个关节全设同一个值，串扰发生了也看不出来）
        #   2) 量级随关节尺寸缩放 → 否则小关节被灌进远大于自身的虚构惯量
        #   3) 必须落在 s800_pace_env_cfg.py 的 armature_bounds / damping_bounds 内
        # 4 组: 0.24(大关节) / 0.14(髋侧摆) / 0.05(中小关节) / 0.008(末端)
        "armature": [
            0.24, 0.14, 0.05, 0.24, 0.05, 0.05,            # 左腿 HIP_P, HIP_R, HIP_Y, KNEE, ANK_P, ANK_R
            0.24, 0.14, 0.05, 0.24, 0.05, 0.05,            # 右腿
            0.05,                                          # 腰 TORSO_YAW
            0.05, 0.05, 0.05, 0.05, 0.008, 0.008, 0.008,   # 左臂 SH_P, SH_R, SH_Y, ELB_P, ELB_Y, WR_P, WR_R
            0.05, 0.05, 0.05, 0.05, 0.008, 0.008, 0.008,   # 右臂
        ],
        # viscous friction（PACE Eq.6 的 d）—— 按电机型号，需与 s800_pace_env_cfg.py 一致
        #   7520-22 (HIP_PITCH/ROLL/KNEE): 1.6   7520-14 (HIP_YAW/TORSO): 1.0
        #   5020 (ANKLE/SHOULDER/ELBOW_PITCH): 0.5   4010 (ELBOW_YAW/WRIST): 0.1
        "damping": [
            1.6, 1.6, 1.0, 1.6, 0.5, 0.5,                  # 左腿
            1.6, 1.6, 1.0, 1.6, 0.5, 0.5,                  # 右腿
            1.0,                                           # 腰
            0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1,             # 左臂
            0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1,             # 右臂
        ],
        # Coulomb friction —— 必须按关节缩放，不能用全场统一值。
        # 教训：曾用 [1.5]*27，结果手腕（最大 PD 力矩 12.5×0.05=0.625Nm）和肘偏航
        # （10.04×0.15=1.506Nm）被 1.5Nm 静摩擦完全锁死，前 3 秒位置几乎不变（1e-5 rad）。
        # 摩擦须明显小于该关节能产生的最大 PD 力矩，否则关节不动、参数不可辨识。
        #   大关节(HIP_PITCH/ROLL/KNEE) PD力矩 ~150-320Nm → 1.0
        #   中小关节                       PD力矩  4-70Nm   → 0.3
        #   末端(ELBOW_YAW/WRIST)          PD力矩 0.6-1.5Nm → 0.05
        "friction": [
            1.0, 1.0, 0.3, 1.0, 0.3, 0.3,                  # 左腿
            1.0, 1.0, 0.3, 1.0, 0.3, 0.3,                  # 右腿
            0.3,                                           # 腰
            0.3, 0.3, 0.3, 0.3, 0.05, 0.05, 0.05,          # 左臂
            0.3, 0.3, 0.3, 0.3, 0.05, 0.05, 0.05,          # 右臂
        ],

        # ── 激励设计：f1_per_joint / directions / bias / scale ─────────────
        # 分组扫频上限：慢组（连杆主导 ω_n≈2.0-2.4Hz）3.0，快组（armature 主导 ω_n≈4.6-5.0Hz）6.0
        # 取值 ≈ 1.5×ω_n；--grouped-sweep 关闭时整列被全局 f1 覆盖
        "f1_per_joint": [
            3.0, 3.0, 6.0, 3.0, 3.0, 6.0,                  # 左腿
            3.0, 3.0, 6.0, 3.0, 3.0, 6.0,                  # 右腿
            3.0,                                           # 腰
            3.0, 3.0, 6.0, 3.0, 6.0, 3.0, 6.0,             # 左臂
            3.0, 3.0, 6.0, 3.0, 6.0, 3.0, 6.0,             # 右臂
        ],
        # 左半为基准 +1，右半按上面的反射判据取符号
        # 校验: python scripts/check_mirror_symmetry.py --robot s800
        "directions": [
            1, 1, 1, 1, 1, 1,                              # 左腿
            1, -1, -1, 1, 1, -1,                           # 右腿
            1,                                             # 腰
            1, 1, 1, 1, 1, 1, 1,                           # 左臂
            1, -1, -1, 1, -1, 1, -1,                       # 右臂
        ],
        # 非对称关节的 URDF 零点（膝 / 踝 roll / 肩 roll / 肘 pitch / 腕 roll）
        "bias": [
            0.000, 0.150, 0.000, 1.046, 0.000, -0.087,     # 左腿
            0.000, 0.150, 0.000, 1.046, 0.000, -0.087,     # 右腿
            0.000,                                         # 腰
            0.000, 1.052, 0.000, -1.004, 0.000, 0.000, -0.131,  # 左臂
            0.000, 1.052, 0.000, -1.004, 0.000, 0.000, -0.131,  # 右臂
        ],
        # 幅度 = min(半行程×0.4, 力矩权限, 物理上限)
        # ⚠️ 幅度直接决定信噪比：曾把 HIP_ROLL 从 0.30 砍到 0.20 → 它是全场最差
        #    （armature 13.6%、viscous 26.3%，第二名的 6 倍）
        "scale": [
            0.500, 0.300, 0.700, 0.500, 0.272, 0.105,      # 左腿
            0.500, 0.300, 0.700, 0.500, 0.272, 0.105,      # 右腿
            0.400,                                         # 腰
            0.350, 0.300, 0.250, 0.300, 0.15, 0.05, 0.05,  # 左臂
            0.350, 0.300, 0.250, 0.300, 0.15, 0.05, 0.05,  # 右臂
        ],

        # ── 全场统一的两项 ────────────────────────────────────────────────
        "bias_gt": 0.05,     # 编码器零位偏差 [rad]
        "delay_gt": 5,       # 指令延时 [仿真步]（5 步 @400Hz = 12.5ms）
    },

    # ══ Unitree G1（29 关节）═══════════════════════════════════════════════
    "g1_sim": {
        # 三项全部取自 assets_robot/g1/g1_29dof.xml 的 <default> 电机类（官方 G1 模型）：
        #   torso/leg/ankle/arm_motor → armature 0.01, damping 0.05, frictionloss 0.2
        #   wrist_motor               → armature 0.01, damping 0.05, frictionloss 0.1
        # 用户明确要求用这份官方 XML，不要换成按电机档的反射惯量（换过，实测更差）。
        "armature": [
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01,            # 左腿
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01,            # 右腿
            0.01, 0.01, 0.01,                              # 腰 yaw/roll/pitch
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,      # 左臂
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,      # 右臂
        ],
        "damping": [
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05,            # 左腿
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05,            # 右腿
            0.05, 0.05, 0.05,                              # 腰
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,      # 左臂
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,      # 右臂
        ],
        # XML 原值：只有腕 **pitch/yaw** 是 wrist_motor 类（frictionloss 0.1，下标 20,21,27,28）；
        # 其余含 wrist_roll 都是 arm_motor 类 0.2 —— wrist_roll 在 XML 里 class="arm_motor"，
        # 别按名字想当然（查过 assets_robot/g1/g1_29dof.xml:274）。
        # ⚠️ 0.1 = 腕峰值力矩(5Nm)的 2%，超过原来 1% 的 bounds 上界 —— 已把 cfg 上界
        #    放宽到 XML 官方值×3（g1_pace_env_cfg.py 的 G1_FRICTION_NOMINAL），
        #    而不是反过来改真值去迁就区间。
        "friction": [
            0.2, 0.2, 0.2, 0.2, 0.2, 0.2,                  # 左腿
            0.2, 0.2, 0.2, 0.2, 0.2, 0.2,                  # 右腿
            0.2, 0.2, 0.2,                                 # 腰
            0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1,             # 左臂（只有腕 pitch/yaw = 0.1）
            0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1,             # 右臂
        ],

        # 分组扫频默认关；开启需按各关节 ω_cl 重设（极点表见 g1_pace_env_cfg.py 注释）
        "f1_per_joint": [
            4.0, 4.0, 4.0, 4.0, 4.0, 4.0,                  # 左腿
            4.0, 4.0, 4.0, 4.0, 4.0, 4.0,                  # 右腿
            4.0, 4.0, 4.0,                                 # 腰
            4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,             # 左臂
            4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,             # 右臂
        ],
        # 由 URDF 世界系转轴按反射判据算出（G1 与 S800 不同，不能照抄）
        # 校验: python scripts/check_mirror_symmetry.py --robot g1
        "directions": [
            1, 1, 1, 1, 1, 1,                              # 左腿（基准）
            1, -1, -1, 1, 1, -1,                           # 右腿 HIP_P, HIP_R, HIP_Y, KNEE, ANK_P, ANK_R
            1, 1, 1,                                       # 腰 yaw/roll/pitch
            1, 1, 1, 1, 1, 1, 1,                           # 左臂（基准）
            1, -1, -1, 1, -1, 1, -1,                       # 右臂 SH_P, SH_R, SH_Y, ELB, WR_R, WR_P, WR_Y
        ],
        # bias = 该关节**行程中点**（对称关节自然为 0）。这样 mid ± 0.4×half_range 恒在行程内
        # —— 若用"站姿"当 bias，hip_roll（左腿 [-0.524, 2.967]，中点 1.222）配 0.698 幅度
        # 在 bias=0 时会扫到 -0.698，超出下限、关节被顶在限位上。
        # 左右取同一个值（镜像由 directions 承担）：
        #   右 hip_roll = -(sin+1.222) ∈ [-1.92, -0.52] ⊂ [-2.967, 0.524] ✓
        # ⚠️ waist_roll/pitch 有**重力塌陷**：kp=28.5 扛不住上半身重量，实测稳态偏差
        #    +0.210(roll) / +0.383(pitch) rad，把实际位置推到行程边缘（waist_roll 实际
        #    [-0.570,+0.470] 越过了 ±0.520 下限，2.5% 时间贴着）。这里按实测**补偿一半**；
        #    pitch 无法全补 —— 全补要 -0.383，配上 ±0.208 幅度后指令自己就到 -0.59 < -0.520。
        #    补偿量与腰 yaw 耦合无关（相关仅 +0.23，是常值重力矩），采完按实测再微调。
        "bias": [
            0.1746, 1.2218, 0.0, 1.3962, -0.1746, 0.0,     # 左腿
            0.1746, 1.2218, 0.0, 1.3962, -0.1746, 0.0,     # 右腿
            0.0, -0.105, -0.19,                            # 腰 yaw/roll/pitch
            -0.2094, 0.3317, 0.0, 0.5236, 0.0, 0.0, 0.0,   # 左臂
            -0.2094, 0.3317, 0.0, 0.5236, 0.0, 0.0, 0.0,   # 右臂
        ],
        # 腿/腰 = 半行程×0.4；臂/腕用更保守的上限（沿用 S800 的实测幅度）
        # ⚠️ 臂的这组上限**没有经过 G1 的力矩/自碰撞校验**：G1 手臂自然下垂时手就贴在髋
        #    外侧，比 S800 更容易自撞（自撞会让腿臂通过接触力耦合，破坏分组优化的前提）。
        #    shoulder_roll 取 0.2 = bias，所以它只往外摆、不会向内扫过身体。
        # waist_yaw 由 1.047 降到 0.5：它扫 ±1.05 rad 时把整个上半身甩起来，是 waist_roll/
        #    pitch 与双臂被甩的共同扰动源；且 1.047 时惯性力矩 ≈85Nm ≈ 自身 effort 上限(88)，
        #    高频段会饱和。S800 的 TORSO_YAW 幅度只有 0.4（且 kp=113，是 G1 腰的 4 倍）。
        "scale": [
            1.082, 0.698, 1.103, 0.593, 0.279, 0.105,      # 左腿
            1.082, 0.698, 1.103, 0.593, 0.279, 0.105,      # 右腿
            0.5, 0.208, 0.208,                             # 腰 yaw/roll/pitch
            0.35, 0.20, 0.15, 0.35, 0.10, 0.10, 0.10,      # 左臂
            0.35, 0.20, 0.15, 0.35, 0.10, 0.10, 0.10,      # 右臂
        ],

        # ── 分组激励（--excite legs / upper）───────────────────────────────
        # 只激励一部分关节时：被选中的关节用这里的幅度和扫频上限，**其余关节停在
        # 各自扫描中心**（bias×dir×scale，即上面那张表的中心姿态），不参与损失。
        #
        # 取值照抄 docs/g1_sim2sim_guide.html 的参考运行（腿部 12 关节、0.1→3Hz、
        # 60 代收敛到 A 档）。⚠️两个必须一起看的前提：
        #   1) 频带要配增益。PACE p.7 要求关节闭环极点 ω_cl 落在激励带内，而 G1 腿部按
        #      当前 BeyondMimic 增益算的 ω_cl 是 hip_p 1.05 / hip_r 1.77 / knee 4.22 /
        #      hip_y 4.58 / ank_p 8.50 / ank_r 9.74 Hz —— 踝的两个是带顶的 3 倍。
        #      那条约束在**真机**上最硬（噪声地板决定能否辨识）；纯 sim2sim 是无噪声确定性
        #      回放，极点出带也常常能辨识出来（参考运行就是低增益 + 3Hz 拿到 A 档）。
        #      所以这轮最该盯的是踝/膝的 armature。要更稳就把 --max_frequency 提到 5，
        #      或按 S800 口径重设腿部增益（ankle 从 28.5 → 7.4，极点 9.7 → 4.97Hz；
        #      hip_pitch 反而要 40 → 150，极点 1.05 → 2.03Hz —— 见 g1_pace_env_cfg.py）。
        #   2) 幅度比全身版小 2~4 倍是刻意的：消掉速度限幅（全身版 knee 有 8.2% 时间
        #      贴在 velocity_limit 上）和轻关节自激。代价是摩擦分辨率下降
        #      （Δτ_f ≲ kp·RMS残差），验收用 cmd/resp 峰峰值比 > 0.3 兜底。
        "excite": {
            "legs": {
                "f1": 3.0,
                "scale": {"HIP_PITCH": 0.275, "HIP_ROLL": 0.20, "HIP_YAW": 0.30,
                          "KNEE": 0.35, "ANKLE_PITCH": 0.25, "ANKLE_ROLL": 0.12},
            },
        },

        "bias_gt": 0.05,     # 编码器零位偏差 [rad]
        # 指令延时 [仿真步]。**不是从 S800 抄的 5，G1 单独取 2**（2 步 @400Hz = 5ms）：
        # 力矩延时在回路内（特征方程 1+C·P·e^{-sT}=0），延时直接吃相位裕度。按 S800 文档
        # 反推的判据 ω_c·T ≲ 0.613 rad，踝的 ω_cl=9.74Hz 在 5 步(12.5ms) 时滞后 43.8° 已超限
        # —— 实测后果是腿部数据里 hip_yaw 54.6%、ankle_pitch 49.4% 的误差能量来自 18.9Hz
        # 的自激极限环，损失地形被搅乱、拟合滑进补偿谷。2 步(5ms) 滞后 17.5°，安全。
        # 参考：docs/g1_sim2sim_guide.html 用指令延时 + 3 步 @2ms=6ms，并写明「力矩延迟在
        # G1 上超过 2 步会失稳」。真要上真机前，用阶跃响应确认实际延时再改这里。
        # ⚠️ 改这个必须同步改 g1_pace_env_cfg.py 的 max_delay 与 delay 上界
        #    （DelayBuffer 只允许 0..max_delay+1，采样越界会直接 ValueError）。
        "delay_gt": 2,
    },
}


# 每个机器人表的数组顺序（= env cfg 的 joint_order）。
# check_mirror_symmetry.py 用它把下标映射回关节名，plot_trajectory 用它把表按名字展开。
JOINT_ORDER = {
    "s800_sim": [
        "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L", "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
        "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R", "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
        "J12_TORSO_YAW",
        "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L", "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L", "J18_WRIST_PITCH_L", "J19_WRIST_ROLL_L",
        "J27_SHOULDER_PITCH_R", "J28_SHOULDER_ROLL_R", "J29_SHOULDER_YAW_R", "J30_ELBOW_PITCH_R", "J31_ELBOW_YAW_R", "J32_WRIST_PITCH_R", "J33_WRIST_ROLL_R",
    ],
    "g1_sim": [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ],
}

# 对照组「全场统一真值」的取值。只有 S800 做过这个实验（26_09_30 那轮）。
# 目的：隔离「搜索维度」这一个变量（ANYmal 的 GT 全场统一 + 四条腿结构相同 ⇒ 损失对
# 腿间参数置换有精确对称性 ⇒ 有效维度远小于 49）。
# ⚠️ 用 --uniform_gt 时必须同时打开 cfg 的 UNIFORM_GT，否则真值落在 bounds 外：
#    S800 现有 14 组分段区间的交集是空集（max(lo)=0.05 > min(hi)=0.02），统一值无处可放。
# ⚠️⚠️ **下面这几个值没有被证实**。26_09_30 那轮是用手改数组的方式注入的（当时的
#    data_collection.py 里没有 uniform 的 CLI），注入值没有任何记录。2026-10-09 用
#    floor test 反查：拿这组值去重放 26_09_30 的数据，损失 3.4e-3（friction 换成 0.1
#    是 3.6e-3），而真值参数本该≈0 —— 说明这组值（至少有一部分）不对。
#    引用这组数做结论之前，先重采一轮 uniform 数据并用 floor test 把真值钉出来。
UNIFORM_CONTROL = {
    "s800_sim": {"armature": 0.05, "viscous": 0.5, "friction": 0.05, "bias": 0.05, "delay": 5},
}


def check_tables() -> list[str]:
    """自检：逐关节数组长度一致、值非负、分组能落位。返回问题列表（空 = 通过）。"""
    problems = []
    for robot, table in ROBOT_TABLES.items():
        order = JOINT_ORDER.get(robot)
        if order is None:
            problems.append(f"{robot}: JOINT_ORDER 里没有对应的关节名列表")
            continue
        for key in TABLE_KEYS:
            if key not in table and key != "excite":     # excite 可选
                problems.append(f"{robot}: 缺键 {key}")
        for key in PER_JOINT_KEYS:
            if key in table and len(table[key]) != len(order):
                problems.append(f"{robot}.{key}: 长度 {len(table[key])} != 关节数 {len(order)}")
        for key in ("armature", "damping", "friction"):
            if any(v < 0 for v in table.get(key, [])):
                problems.append(f"{robot}.{key}: 有负值")
        for i, name in enumerate(order):
            try:
                joint_group(name)
            except ValueError as e:
                problems.append(f"{robot}[{i}]: {e}")
        # excite 里引用的关节类型必须真实存在
        for gname, cfg in table.get("excite", {}).items():
            types = {joint_type(n) for n in order}
            unknown = set(cfg.get("scale", {})) - types
            if unknown:
                problems.append(f"{robot}.excite[{gname}].scale 有未知关节类型 {sorted(unknown)}")
    return problems


def describe(robot: str) -> str:
    """逐关节展开成一张可读的表（每行一个关节，grep 关节名即可）。"""
    table = ROBOT_TABLES[robot]
    order = JOINT_ORDER[robot]
    head = (f"{'#':>3} {'关节':28s} {'armature':>9s} {'damping':>8s} {'friction':>9s}"
            f" {'dir':>4s} {'bias':>8s} {'scale':>7s} {'f1':>5s}")
    lines = [f"# {robot}  ({len(order)} 关节)  顺序 = env cfg 的 joint_order", head, "#" + "-" * (len(head) + 1)]
    for i, name in enumerate(order):
        lines.append(f"{i:3d} {name:28s} {table['armature'][i]:9.4f} {table['damping'][i]:8.3f} "
                     f"{table['friction'][i]:9.4f} {table['directions'][i]:+4d} "
                     f"{table['bias'][i]:8.4f} {table['scale'][i]:7.3f} {table['f1_per_joint'][i]:5.1f}")
    lines.append(f"# 全场统一: bias_gt={table['bias_gt']} rad   delay_gt={table['delay_gt']} 步")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="按关节名展开逐机器人参数表（找参数用这个）")
    ap.add_argument("--robot", default=None,
                    help=f"{sorted(ROBOT_TABLES)}，也可写 g1 / s800（与 check_mirror_symmetry.py 一致）")
    args = ap.parse_args()

    # g1 -> g1_sim 之类的简写（check_mirror_symmetry.py 用的是短名）
    _alias = {k.rsplit("_", 1)[0]: k for k in ROBOT_TABLES}
    if args.robot:
        args.robot = _alias.get(args.robot, args.robot)
        if args.robot not in ROBOT_TABLES:
            raise SystemExit(f"✗ 未知机器人 {args.robot}，可用: {sorted(ROBOT_TABLES)}（或 {sorted(_alias)}）")

    problems = check_tables()
    for p in problems:
        print(f"⚠️ {p}")
    for r in ([args.robot] if args.robot else sorted(ROBOT_TABLES)):
        print(describe(r))
        print()
    sys.exit(1 if problems else 0)
