"""Per-run hip hold gains for knee-only excitation (no simulator imports)."""
import math
import re


def apply_hip_hold_gains(actuator, joint_names, mode, kp=None, kd=None):
    if kp is None and kd is None:
        return
    if mode != 'knee_pitch':
        raise ValueError('Hip hold gain overrides require --leg_sweep knee_pitch')
    updates = {}
    for attr, values in (('stiffness', kp), ('damping', kd)):
        if values is None:
            continue
        if len(values) != 3 or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Hip gains must be three finite nonnegative values: pitch roll yaw')
        original = getattr(actuator, attr)
        resolved = {}
        for name in joint_names:
            if isinstance(original, dict):
                matches = [value for pattern, value in original.items() if re.fullmatch(pattern, name)]
                if len(matches) != 1:
                    raise ValueError(f'Ambiguous or missing {attr} for {name}')
                resolved[name] = matches[0]
            else:
                resolved[name] = original
        for kind, value in zip(('HIP_PITCH', 'HIP_ROLL', 'HIP_YAW'), values):
            pair = [n for n in joint_names if n[4:-2] == kind and n.endswith(('_L', '_R'))]
            if len(pair) != 2:
                raise ValueError(f'Expected left/right {kind}')
            for name in pair:
                resolved[name] = float(value)
        updates[attr] = resolved
    for attr, values in updates.items():
        setattr(actuator, attr, values)
