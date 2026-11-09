"""S800 stand/fixture obstacle model for data-collection simulations.

The fixture is a *collision obstacle*, not a support model: the robot base stays
fixed by ``fix_root_link`` and the fixture never carries load.  Geometry is
described in the robot base-link frame (+x forward, +y left, +z up), so measured
dimensions can be entered as "distance below/behind the pelvis".

Pure spec handling and contact summaries live here and stay importable without
Isaac Sim.  Scene helpers import IsaacLab lazily.
"""

from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path

import numpy as np

DEFAULT_SPEC_PATH = Path(__file__).with_name("fixtures") / "s800_stand_placeholder.json"
DEFAULT_COLOR = (0.35, 0.35, 0.35)


def default_fixture_path() -> Path:
    return DEFAULT_SPEC_PATH


def _triplet(value, name, *, positive=False, unit_interval=False):
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite 3-vector, got {value!r}")
    if positive and (arr <= 0).any():
        raise ValueError(f"{name} must be positive, got {value!r}")
    if unit_interval and ((arr < 0).any() or (arr > 1).any()):
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return arr


def validate_fixture_spec(spec):
    """Validate a fixture JSON dict and return a normalized deep copy."""
    if not isinstance(spec, dict):
        raise ValueError("Fixture spec must be a JSON object")
    parts = spec.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Fixture spec needs a nonempty 'parts' list")
    normalized = copy.deepcopy(spec)
    normalized.setdefault("frame", "robot_base")
    names = set()
    for part in normalized["parts"]:
        if not isinstance(part, dict):
            raise ValueError("Each fixture part must be an object")
        name = part.get("name")
        if not isinstance(name, str) or not name or not name.replace("_", "").isalnum():
            raise ValueError(f"Fixture part name must be alphanumeric/underscore, got {name!r}")
        if name in names:
            raise ValueError(f"Duplicate fixture part name: {name}")
        names.add(name)
        shape = part.get("shape")
        if shape == "box":
            _triplet(part.get("size"), f"{name}.size", positive=True)
        elif shape == "cylinder":
            for key in ("radius", "height"):
                value = part.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name}.{key} must be a positive finite number")
        else:
            raise ValueError(f"{name}.shape must be 'box' or 'cylinder', got {shape!r}")
        part["center"] = _triplet(part.get("center"), f"{name}.center").tolist()
        part["rpy"] = _triplet(part.get("rpy", (0., 0., 0.)), f"{name}.rpy").tolist()
        part["color"] = _triplet(part.get("color", DEFAULT_COLOR), f"{name}.color",
                                 unit_interval=True).tolist()
        collision = part.setdefault("collision", True)
        if not isinstance(collision, bool):
            raise ValueError(f"{name}.collision must be true or false")
    return normalized


def load_fixture_spec(path=None):
    """Load and validate a fixture JSON; ``None`` uses the packaged placeholder."""
    source = Path(path) if path is not None else DEFAULT_SPEC_PATH
    try:
        data = json.loads(source.read_text())
    except OSError as exc:
        raise ValueError(f"Cannot read fixture spec {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in fixture spec {source}: {exc}") from exc
    return validate_fixture_spec(data)


def quat_from_rpy(rpy):
    """URDF fixed-axis roll/pitch/yaw to (w, x, y, z)."""
    r, p, y = (float(v) for v in rpy)
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def quat_rotate(q, vector):
    q = np.asarray(q, dtype=float)
    v = np.asarray(vector, dtype=float)
    if q.shape != (4,) or v.shape != (3,):
        raise ValueError("quat_rotate expects quaternion (w,x,y,z) and 3-vector")
    q = q / np.linalg.norm(q)
    w, xyz = q[0], q[1:]
    return v + 2.0 * (w * np.cross(xyz, v) + np.cross(xyz, np.cross(xyz, v)))


def part_world_pose(part, base_pos, base_rot=(1., 0., 0., 0.)):
    """Fixture part pose in world frame from a base-frame spec."""
    world_pos = np.asarray(base_pos, dtype=float) + quat_rotate(base_rot, part["center"])
    world_rot = quat_mul(base_rot, quat_from_rpy(part["rpy"]))
    return tuple(float(v) for v in world_pos), tuple(float(v) for v in world_rot)


def add_fixture_to_scene(scene_cfg, spec, base_pos, base_rot=(1., 0., 0., 0.),
                         root="/World/s800_fixture"):
    """Attach static collision/visual prims to an InteractiveSceneCfg.

    Call before ``gym.make``.  Returns the spawned prim paths for logging.
    """
    from isaaclab.assets import AssetBaseCfg
    import isaaclab.sim as sim_utils

    prim_paths = []
    for part in spec["parts"]:
        pos, rot = part_world_pose(part, base_pos, base_rot)
        common = dict(
            collision_props=(sim_utils.CollisionPropertiesCfg() if part["collision"] else None),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(part["color"])),
        )
        if part["shape"] == "box":
            spawn = sim_utils.CuboidCfg(size=tuple(part["size"]), **common)
        else:
            spawn = sim_utils.CylinderCfg(radius=part["radius"], height=part["height"],
                                          axis="Z", **common)
        attr = f"fixture_{part['name']}"
        if attr in scene_cfg.__dict__:
            raise ValueError(f"Scene config already has a field named {attr}")
        setattr(scene_cfg, attr, AssetBaseCfg(
            prim_path=f"{root}/{part['name']}", spawn=spawn,
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
        ))
        prim_paths.append(f"{root}/{part['name']}")
    return prim_paths


def add_contact_sensor(scene_cfg, name="fixture_contact"):
    """Add a whole-robot contact sensor; requires spawn.activate_contact_sensors=True."""
    from isaaclab.sensors import ContactSensorCfg

    if name in scene_cfg.__dict__:
        raise ValueError(f"Scene config already has a field named {name}")
    setattr(scene_cfg, name, ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0.0,
        history_length=0, debug_vis=False, track_air_time=False,
    ))


def fixture_joint_limit_violation(q, joint_names, margin=0.0):
    """Check the measured stand's effective joint ranges.

    The stand leaves no positive HIP_PITCH (backward swing) and no inward HIP_ROLL:
    left HIP_ROLL >= 0, right HIP_ROLL <= 0.  ``q`` must be in the commanded-target
    space (encoder bias excluded), shape ``(samples, joints)``.
    """
    if hasattr(q, "detach"):
        q = q.detach().cpu().numpy()
    q = np.asarray(q, dtype=float)
    names = list(joint_names)
    if q.ndim != 2 or q.shape[1] != len(names) or not len(q) or not np.isfinite(q).all():
        raise ValueError("q must be a finite (samples, joints) array matching joint_names")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Joint-range margin must be finite and nonnegative")
    limits = {}
    worst = None
    for i, name in enumerate(names):
        upper = name.upper()
        if "HIP_PITCH" in upper:
            violation = max(0.0, float(q[:, i].max()) - margin)
            limits[name] = {"min_rad": float(q[:, i].min()), "max_rad": float(q[:, i].max()),
                            "allowed": "q <= 0", "violation_rad": violation}
            direction = "positive/backward"
        elif "HIP_ROLL" in upper:
            outward_sign = -1.0 if upper.endswith("_R") else 1.0
            outward = outward_sign * q[:, i]
            violation = max(0.0, -float(outward.min()) - margin)
            limits[name] = {"min_rad": float(q[:, i].min()), "max_rad": float(q[:, i].max()),
                            "allowed": "q >= 0" if outward_sign > 0 else "q <= 0",
                            "violation_rad": violation}
            direction = "inward"
        else:
            continue
        if worst is None or violation > worst["violation_rad"]:
            worst = {"joint": name, "direction": direction, "violation_rad": violation}
    return {"passed": worst is None or worst["violation_rad"] <= 1e-12,
            "margin_rad": float(margin), "joints": limits, "worst": worst,
            "scope": "measured stand: HIP_PITCH<=0; HIP_ROLL outward only"}


def summarize_contact(force_norm, times, body_names, threshold, dt=None, ignore_bodies=()):
    """JSON-safe summary of a (T, B) contact-force-norm array.

    ``force_norm[i]`` is the contact force during the step started at ``times[i]``.
    ``ignore_bodies`` removes expected support contacts (e.g. LINK_BASE on the
    column top) from detection and peaks, but the CSV keeps every body.
    """
    force = np.asarray(force_norm, dtype=float)
    times = np.asarray(times, dtype=float)
    names = list(body_names)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Contact threshold must be finite and nonnegative")
    if force.ndim != 2 or force.shape[0] != len(times) or force.shape[1] != len(names):
        raise ValueError("force_norm must be (len(times), len(body_names))")
    if not np.isfinite(force).all() or (force < 0).any():
        raise ValueError("Contact forces must be finite and nonnegative")
    ignored = [i for i, name in enumerate(names) if name in set(ignore_bodies)]
    keep = np.ones(len(names), dtype=bool)
    keep[ignored] = False
    checked = force[:, keep]
    active = checked > threshold
    detected = bool(active.any())
    summary = {
        "contact_detected": detected,
        "threshold_N": float(threshold),
        "max_force_N": float(checked.max()) if checked.size else 0.0,
        "body_names": names,
        "ignored_bodies": [names[i] for i in ignored],
        "samples": int(len(times)),
        "scope": "net contact force per robot body; includes fixture, ground and self contacts",
    }
    if detected:
        frame, reduced_body = np.unravel_index(int(np.argmax(active)), active.shape)
        kept_indices = np.flatnonzero(keep)
        body = int(kept_indices[reduced_body])
        peak_by_body = checked.max(axis=0)
        order = np.argsort(peak_by_body)[::-1]
        summary.update(
            first_contact_time_s=float(times[frame]),
            first_contact_body=names[body],
            first_contact_force_N=float(force[frame, body]),
            peak_by_body_N={names[int(kept_indices[i])]: float(peak_by_body[i])
                            for i in order if peak_by_body[i] > threshold},
        )
        if len(times) > 1:
            step = float(dt if dt is not None else times[1] - times[0])
            summary["contact_duration_s"] = float(active.any(axis=1).sum() * step)
    return summary


def write_contact_csv(path, times, force_norm, body_names):
    """Write the per-body contact-force curve next to the recording."""
    force = np.asarray(force_norm, dtype=float)
    times = np.asarray(times, dtype=float)
    names = list(body_names)
    if force.ndim != 2 or force.shape != (len(times), len(names)):
        raise ValueError("force_norm must be (len(times), len(body_names))")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", *names])
        for time, row in zip(times, force):
            writer.writerow([f"{float(time):.6f}", *(f"{float(v):.6f}" for v in row)])
