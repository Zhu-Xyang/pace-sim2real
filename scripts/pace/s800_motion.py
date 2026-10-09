"""S800 excitation centers and signed amplitudes, both in radians."""

import numpy as np
import torch

from s800_clearance import with_midpoints


def nominal_motion(joint_names):
    # Explicit centers preserve the former bias*direction*scale pose. Changing an
    # amplitude below no longer moves the center. Encoder bias is a separate value.
    centers = {"HIP_PITCH": 0., "HIP_ROLL": .045, "HIP_YAW": 0., "KNEE_PITCH": .523,
               "ANKLE_PITCH": 0., "ANKLE_ROLL": -.009135,
               "SHOULDER_PITCH": 0., "SHOULDER_ROLL": .3156, "SHOULDER_YAW": 0.,
               "ELBOW_PITCH": -.3012, "ELBOW_YAW": 0., "WRIST_PITCH": 0., "WRIST_ROLL": -.00655}
    amplitudes = {"HIP_PITCH": .5, "HIP_ROLL": .3, "HIP_YAW": .7, "KNEE_PITCH": .5,
                  "ANKLE_PITCH": .272, "ANKLE_ROLL": .105,
                  "SHOULDER_PITCH": .35, "SHOULDER_ROLL": .3, "SHOULDER_YAW": .25,
                  "ELBOW_PITCH": .3, "ELBOW_YAW": .15, "WRIST_PITCH": .05, "WRIST_ROLL": .05}
    mirrored = {"HIP_ROLL", "HIP_YAW", "ANKLE_ROLL", "SHOULDER_ROLL", "SHOULDER_YAW", "ELBOW_YAW", "WRIST_ROLL"}
    center, amplitude = [], []
    for name in joint_names:
        kind = name[4:-2] if name.endswith(("_L", "_R")) else name[4:]
        if kind == "TORSO_YAW":
            center.append(0.)
            amplitude.append(.4)
        else:
            sign = -1 if name.endswith("_R") and kind in mirrored else 1
            center.append(sign * centers[kind])
            amplitude.append(sign * amplitudes[kind])
    return torch.tensor(center), torch.tensor(amplitude)


def scaled_nominal_motion(joint_names, selected, factor):
    """Restore (chirp + original_bias) * direction * (original_scale * factor).

    Scale both the center and amplitude for excited joints only. This preserves
    the original normalized trajectory bias, not the later wide-stance pose.
    Encoder bias is independent and must not be changed here.
    """
    if not np.isfinite(factor) or not 0 < factor <= 1:
        raise ValueError("scale_factor must be finite and in (0, 1]")
    center, amplitude = nominal_motion(joint_names)
    center[selected] *= factor
    amplitude[selected] *= factor
    return center, amplitude


def choose_leg_motion(checker, wave, center, amplitude, encoder_bias, selected, margin):
    """Search roll centers/roll-yaw amplitudes; verify finalists at every sample + midpoint.

    Screening samples only rank candidates. Acceptance always uses the full dense
    trajectory. The reserve above the requested margin is for tracking error,
    but actual recorded motion must independently pass the requested margin.
    """
    wave = np.asarray(wave, dtype=float)
    center, amplitude = np.array(center, dtype=float), np.array(amplitude, dtype=float)
    encoder_bias = np.asarray(encoder_bias, dtype=float)
    names = checker.names
    roll = [names.index(n) for n in ("J01_HIP_ROLL_L", "J07_HIP_ROLL_R")]
    yaw = [names.index(n) for n in ("J02_HIP_YAW_L", "J08_HIP_YAW_R")]
    candidates = []
    reserve = margin + .03
    sparse = wave[::max(1, len(wave)//300)]
    if not set(roll+yaw) <= set(selected):
        candidates = [(0., center, amplitude)]
    else:
        # The legacy full-amplitude sweep hits leg limits during dynamic replay.
        # Start at 60% amplitude; actual-motion acceptance remains mandatory.
        leg_indices = [i for i, n in enumerate(names) if any(k in n for k in ("HIP_", "KNEE_", "ANKLE_"))]
        amplitude[leg_indices] *= .6
        for i, n in enumerate(names):
            if "ANKLE_ROLL" in n:
                amplitude[i] *= .5
        # The smaller geometrically feasible poses still let the feet approach
        # under asymmetric tracking error in the 0.1--8 Hz dynamic replay.
        for c in (.6, .65):
            for ar in (.18, .12, .09, .06):
                # Hip yaw overshoots strongly near the low-frequency resonance.
                for ay in (.21, .15, .105, .06):
                    cc, aa = center.copy(), amplitude.copy()
                    cc[roll], aa[roll], aa[yaw] = [c, -c], [ar, -ar], [ay, -ay]
                    if checker.check(cc+sparse*aa+encoder_bias, margin=reserve)["passed"]:
                        # Prefer preserving excitation, then the smaller outward center.
                        candidates.append((ar/.3 + ay/.7 - .5*c, cc, aa))
    for score, cc, aa in sorted(candidates, key=lambda item: item[0], reverse=True):
        report = checker.check(with_midpoints(cc+wave*aa+encoder_bias), margin=reserve)
        if report["passed"]:
            report.update(planning_reserve_m=.03, midpoint_checks=True,
                          continuous_collision_certificate=False)
            return torch.tensor(cc, dtype=torch.float32), torch.tensor(aa, dtype=torch.float32), report
    raise ValueError("No leg trajectory satisfies the geometry/limit margins. Reduce excitation or redesign the center pose.")
