# ═══════════════════════════════════════════════════════════════════════
# Joint names (27 DOF, matches URDF joint order)
# ═══════════════════════════════════════════════════════════════════════
S800_JOINT_NAMES = [
    # Left leg
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
    # Right leg
    "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
    # Torso
    "J12_TORSO_YAW",
    # Left arm
    "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L",
    "J18_WRIST_PITCH_L", "J19_WRIST_ROLL_L",
    # Right arm
    "J27_SHOULDER_PITCH_R", "J28_SHOULDER_ROLL_R", "J29_SHOULDER_YAW_R",
    "J30_ELBOW_PITCH_R", "J31_ELBOW_YAW_R",
    "J32_WRIST_PITCH_R", "J33_WRIST_ROLL_R",
]  # fmt: skip

def _joint_type(joint_name: str) -> str:
    """Extract joint type from joint name, e.g. 'J00_HIP_PITCH_L' -> 'HIP_PITCH'."""
    stripped = joint_name[4:]               # remove 'J00_' prefix
    if stripped.endswith("_L") or stripped.endswith("_R"):
        stripped = stripped[:-2]           # remove '_L' / '_R' suffix
    return stripped

# Per joint-type physical data (URDF; armature from the shared T800 actuator
# types) and HYBRID design values (tuned gains + reduced arm scale; see
# docs/action_scale_design.md section 5). Wrist limits are the vendor URDF
# values -- see the placeholder warning in the module docstring.
_TYPE_TABLE = {
    #                 tau_lim  v_lim  armature      kp     kd    scale
    "HIP_PITCH":      (415.0,  25.96, 0.2427264,   240.0,  5.8,  0.43),
    "HIP_ROLL":       (370.0,  25.31, 0.14110848,  200.0,  4.2,  0.46),
    "HIP_YAW":        (222.0,  23.19, 0.0448737,   113.0,  3.2,  0.49),
    "KNEE_PITCH":     (415.0,  25.96, 0.2427264,   240.0,  5.8,  0.43),
    "ANKLE_PITCH":    (160.0,  33.51, 0.0354625,    90.0,  0.45, 0.45),
    "ANKLE_ROLL":     (160.0,  33.51, 0.0354625,    90.0,  0.45, 0.15),  # range-capped
    "TORSO_YAW":      (222.0,  23.19, 0.0448737,   113.0,  3.2,  0.49),
    "SHOULDER_PITCH": (160.0,  33.51, 0.0354625,    90.0,  0.4,  0.20),  # hybrid arm
    "SHOULDER_ROLL":  (160.0,  33.51, 0.0354625,    90.0,  0.4,  0.20),  # hybrid arm
    "SHOULDER_YAW":   (160.0,  33.51, 0.0354625,    90.0,  0.4,  0.20),  # hybrid arm
    "ELBOW_PITCH":    (160.0,  33.51, 0.0354625,    90.0,  0.4,  0.20),  # hybrid arm
    "ELBOW_YAW":      (52.0,   35.2,  0.00671625,   50.0,  0.3,  0.12),  # hybrid arm
    "WRIST_PITCH":    (5.0,    150.0, 0.005,        12.5,  0.13, 0.10),  # tau from soma-retarget URDF
    "WRIST_ROLL":     (5.0,    150.0, 0.005,        12.5,  0.13, 0.10),  # tau from soma-retarget URDF
}  # fmt: skip

S800_EFFORT_LIMITS = {n: _TYPE_TABLE[_joint_type(n)][0] for n in S800_JOINT_NAMES}
S800_VELOCITY_LIMITS = {n: _TYPE_TABLE[_joint_type(n)][1] for n in S800_JOINT_NAMES}
S800_ARMATURE = {n: _TYPE_TABLE[_joint_type(n)][2] for n in S800_JOINT_NAMES}
S800_HYBRID_STIFFNESS = {n: _TYPE_TABLE[_joint_type(n)][3] for n in S800_JOINT_NAMES}
S800_HYBRID_DAMPING = {n: _TYPE_TABLE[_joint_type(n)][4] for n in S800_JOINT_NAMES}
S800_HYBRID_ACTION_SCALE = {n: _TYPE_TABLE[_joint_type(n)][5] for n in S800_JOINT_NAMES}

# Deployment-style rest pose carried over from the T800 family (identical body
# kinematics conventions); wrists default to zero. All values are inside the
# S800 joint ranges.
S800_DEFAULT_JOINT_POS = {
    "J00_HIP_PITCH_L": -0.06, "J03_KNEE_PITCH_L": 0.12, "J04_ANKLE_PITCH_L": -0.06,
    "J06_HIP_PITCH_R": -0.06, "J09_KNEE_PITCH_R": 0.12, "J10_ANKLE_PITCH_R": -0.06,
    "J14_SHOULDER_ROLL_L": 0.15, "J16_ELBOW_PITCH_L": -0.25,
    "J28_SHOULDER_ROLL_R": -0.15, "J30_ELBOW_PITCH_R": -0.25,
}  # fmt: skip
