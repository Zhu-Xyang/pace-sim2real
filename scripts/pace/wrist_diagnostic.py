"""26-DoF wrist-hold diagnostic helpers; no Isaac Sim imports or launch."""

import csv
import hashlib
import math
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
URDF_DIR = ROOT / "assets_robot/robot_s/urdf"
URDF = URDF_DIR / "serial_robot_s_pace_26dof.urdf"
USD = ROOT / "assets_robot/robot_s/usd_wrist_diagnostic/serial_robot_s_pace_26dof.usda"
WRISTS = ("J18_WRIST_PITCH_L", "J19_WRIST_ROLL_L", "J32_WRIST_PITCH_R", "J33_WRIST_ROLL_R")


def prepare_urdf():
    """Unlock only four wrists in the current 22-DoF model, preserving everything else."""
    source = URDF_DIR / "serial_robot_s_pace_22dof.urdf"
    original = ET.parse(URDF_DIR / "serial_robot_s_42dof.urdf").getroot()
    text = source.read_text()
    before = ET.fromstring(text)
    expected = deepcopy(before)
    for name in WRISTS:
        joint = expected.find(f"joint[@name='{name}']")
        full = original.find(f"joint[@name='{name}']")
        if joint is None or full is None or joint.get("type") != "fixed" or full.get("type") != "revolute":
            raise ValueError(f"Unexpected wrist definition: {name}")
        joint.set("type", "revolute")
        if ET.tostring(joint) != ET.tostring(full):
            raise ValueError(f"Locked/full URDF differs beyond joint type: {name}")
        text, count = re.subn(rf'(<joint\s+name="{name}"\s+type=")fixed(")', r'\1revolute\2', text)
        if count != 1:
            raise ValueError(f"Expected one declaration for {name}")
    after = ET.fromstring(text)
    if ET.tostring(after) != ET.tostring(expected):
        raise ValueError("Unexpected URDF change")
    movable = [j.get("name") for j in after.findall("joint") if j.get("type") != "fixed"]
    if len(movable) != 26:
        raise ValueError(f"Expected 26 moving joints, got {len(movable)}")
    URDF.write_text(text)
    mass = sum(float(x.get("value")) for x in after.findall("link/inertial/mass"))
    print(f"[prepared] {URDF}\n26 moving joints; mass unchanged: {mass:.8f} kg; only wrist joint types changed.")


def configure(env_cfg, kp, kd):
    """Opt-in collector configuration; leave the default fit/22-DoF config intact."""
    if not all(math.isfinite(v) and v >= 0 for v in (kp, kd)):
        raise ValueError("wrist_kp and wrist_kd must be finite and nonnegative")
    if not USD.is_file() or not URDF.is_file():
        raise FileNotFoundError(f"Prepare URDF and convert first: {USD}")
    old = list(env_cfg.sim2real.joint_order)
    if len(old) != 22 or set(old) & set(WRISTS):
        raise ValueError("Wrist diagnostic expects the original 22-DoF configuration")
    names = sorted(old + list(WRISTS))
    root = ET.parse(URDF).getroot()
    moving = {j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"}
    if moving != set(names):
        raise ValueError("26-DoF URDF and requested joint set differ")
    limits = {n: float(root.find(f"joint[@name='{n}']/limit").get("effort")) for n in names}
    cfg = env_cfg.scene.robot
    cfg.spawn.usd_path = str(USD)
    base = deepcopy(cfg.actuators["joints"])
    base.joint_names_expr = old
    wrist = deepcopy(base)
    wrist.joint_names_expr = list(WRISTS)
    wrist.stiffness, wrist.damping = kp, kd
    wrist.effort_limit = {n: limits[n] for n in WRISTS}
    wrist.velocity_limit = {n: float(root.find(f"joint[@name='{n}']/limit").get("velocity")) for n in WRISTS}
    # Separate actuator avoids using the hip's 415 Nm torque-speed intercept for wrists.
    if len({limits[n] for n in WRISTS}) != 1:
        raise ValueError("Split wrist actuator groups if wrist torque limits differ")
    wrist.saturation_effort = limits[WRISTS[0]]
    wrist.viscous_friction = 0.1
    wrist.friction = wrist.dynamic_friction = 0.05
    cfg.actuators = {"joints": base, "wrists": wrist}
    n, m = len(names), len(old)
    bounds = torch.zeros(4 * n + 1, 2)
    for block in range(4):
        for j, name in enumerate(names):
            bounds[block * n + j] = (env_cfg.sim2real.bounds_params[block * m + old.index(name)]
                                      if name in old else torch.tensor(((0.001, .03), (0., .5), (0., .1), (-.1, .1))[block]))
    bounds[-1] = env_cfg.sim2real.bounds_params[-1]
    env_cfg.sim2real.joint_order = names
    env_cfg.sim2real.bounds_params = bounds
    return {"mode": "wrist_hold_26dof", "wrist_names": list(WRISTS), "wrist_kp": kp, "wrist_kd": kd,
            "wrist_target_rad": 0., "urdf_effort_limit": torch.tensor([limits[n] for n in names]),
            "limit_source": str(URDF), "urdf_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest(),
            "limitations": "URDF limits are a reference, not a captured SDK runtime GetTorqueLimit result. "
                           "Synthetic dynamics; starts at target, no hardware entry/gain transition replay. "
                           "Torso/head/fingers remain mechanically fixed; fixture not modeled."}


def predicted_pd(target, position, velocity, kp, kd):
    """SDK-style pre-publish guard (qd_des=0, tau_ff=0), without command delay."""
    return kp * (target - position) - kd * velocity


def save_torque_report(output, time, names, requested, computed, applied, kp, kd, sim_limit, reference_limit):
    """Save every sample, plus peaks/exceedance summaries, including held wrists."""
    arrays = [x.detach().cpu() for x in (time, requested, computed, applied, kp, kd, sim_limit, reference_limit)]
    time, requested, computed, applied, kp, kd, sim_limit, reference_limit = arrays
    if not all(torch.isfinite(a).all() for a in arrays):
        raise ValueError("Nonfinite torque diagnostics; cannot produce a valid limit report")
    with output.with_suffix(".torque.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s"] + [f"{label}/{name}" for label in ("pd_requested", "pd_computed", "applied") for name in names])
        writer.writerows(torch.cat([time[:, None], requested, computed, applied], dim=1).tolist())
    rows = []
    for i, name in enumerate(names):
        over = requested[:, i].abs() > reference_limit[i]
        idx = torch.nonzero(over).flatten()
        rows.append(dict(joint=name, kp=float(kp[i]), kd=float(kd[i]),
                         sim_effort_limit_nm=float(sim_limit[i]), urdf_reference_limit_nm=float(reference_limit[i]),
                         requested_peak_nm=float(requested[:, i].abs().max()),
                         computed_peak_nm=float(computed[:, i].abs().max()),
                         applied_peak_nm=float(applied[:, i].abs().max()),
                         reference_exceed_percent=100 * float(over.float().mean()),
                         first_reference_exceed_s=float(time[idx[0]]) if len(idx) else None,
                         computed_over_sim_limit_percent=100 * float((computed[:, i].abs() > sim_limit[i]).float().mean())))
    with output.with_suffix(".torque_summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("[torque] joint: requested / computed / applied peak [Nm]; URDF reference; exceed %")
    for r in rows:
        print(f"{r['joint']:<24} {r['requested_peak_nm']:9.4f} / {r['computed_peak_nm']:9.4f} / "
              f"{r['applied_peak_nm']:9.4f}; {r['urdf_reference_limit_nm']:7.2f}; {r['reference_exceed_percent']:7.3f}%")
    return rows


if __name__ == "__main__":
    # Only writes XML; does not launch Isaac Sim or run any dynamics.
    prepare_urdf()
