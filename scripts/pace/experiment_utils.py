"""Shared grouping and recording checks; importable without starting Isaac Sim."""

import torch


def build_joint_groups(joint_order):
    groups = {"legs": [], "torso": [], "arms": []}
    for i, name in enumerate(joint_order):
        upper = name.upper()
        if any(k in upper for k in ("HIP_", "KNEE", "ANKLE_")):
            group = "legs"
        elif any(k in upper for k in ("TORSO", "WAIST")):
            group = "torso"
        elif any(k in upper for k in ("SHOULDER_", "ELBOW_", "WRIST_")):
            group = "arms"
        else:
            raise ValueError(f"Unrecognized joint: {name}")
        groups[group].append(i)
    return groups


def select_groups(group, joint_order):
    groups = build_joint_groups(joint_order)
    names = [g.strip() for g in group.split(",") if g.strip()]
    if names == ["all"]:
        return None, list(range(len(joint_order)))
    if not names or set(names) - groups.keys():
        raise ValueError("group must be all, legs, torso, arms, or comma-separated groups")
    names = [g for g in groups if g in names]
    indices = sorted(i for g in names for i in groups[g])
    if not indices:
        raise ValueError("Selected group has no joints")
    return names, indices


def recording_name(group, delay_mode):
    return "chirp_data.pt" if group == "all" else f"chirp_{group.replace(',', '_')}_{delay_mode}.pt"


def configure_delay(env_cfg, mode):
    from pace_sim2real.utils.pace_actuator import PaceDCMotor, PaceDCMotorCmdDelay
    for actuator in env_cfg.scene.robot.actuators.values():
        actuator.class_type = PaceDCMotorCmdDelay if mode == "command" else PaceDCMotor


def group_chirp(dt, duration, rest, ramp, f0, f1, center, amplitude, selected):
    """All arrays use config joint order. Rest -> tapered linear chirp -> rest."""
    if dt <= 0 or duration <= 0 or rest < 0 or ramp < 0 or 2 * ramp > duration:
        raise ValueError("Require dt,duration>0, rest>=0 and 0<=2*ramp<=duration")
    if f0 <= 0 or torch.any(f1 < f0) or torch.any(f1 >= 0.5 / dt):
        raise ValueError("Chirp frequencies must satisfy 0<f0<=f1<Nyquist")
    n_rest, n_sweep = round(rest / dt), round(duration / dt)
    if n_sweep < 2:
        raise ValueError("Chirp needs at least two samples")
    time = torch.arange(n_sweep + 2 * n_rest, device=center.device) * dt
    t = torch.arange(n_sweep, device=center.device) * dt
    phase = 2 * torch.pi * (f0 * t[:, None] + (f1 - f0) * t[:, None] ** 2 / (2 * n_sweep * dt))
    window = torch.ones_like(t)
    if ramp:
        edge = torch.minimum(t, t[-1] - t).clamp(0, ramp) / ramp
        window = 0.5 - 0.5 * torch.cos(torch.pi * edge)
    commands = center.repeat(len(time), 1)
    commands[n_rest:n_rest + n_sweep, selected] += (
        window[:, None] * torch.sin(phase[:, selected]) * amplitude[selected]
    )
    return time, commands


def truth_vector(data, joint_order):
    gt = data.get("gt")
    if gt is None or list(gt["joint_order"]) != list(joint_order):
        raise ValueError("This diagnostic needs recorded gt in the same joint order")
    return torch.cat([torch.as_tensor(gt[k]).cpu().flatten() for k in
                      ("armature", "damping", "friction", "bias")] +
                     [torch.tensor([float(gt["delay"])])])


def freeze_unfitted_truth(bounds, data, joint_order, selected, warm_start=None):
    """Sim-to-sim diagnostic: active parameters never use their truth as initialization."""
    result = bounds.mean(dim=1).cpu() if warm_start is None else warm_start.cpu().clone()
    truth = truth_vector(data, joint_order)
    n = len(joint_order)
    frozen = [j + block * n for block in range(4) for j in range(n) if j not in selected]
    result[frozen] = truth[frozen]
    if torch.any(result[frozen] < bounds.cpu()[frozen, 0]) or torch.any(result[frozen] > bounds.cpu()[frozen, 1]):
        raise ValueError("Frozen truth lies outside bounds; correct bounds before fitting")
    return result


def actuator_metadata(env, cfg, joint_ids, delay_mode):
    robot = env.unwrapped.scene["robot"]
    values = {key: torch.zeros(robot.num_joints) for key in ("kp", "kd", "effort_limit", "velocity_limit")}
    for actuator in robot.actuators.values():
        for key, attr in (("kp", "stiffness"), ("kd", "damping"),
                          ("effort_limit", "effort_limit"), ("velocity_limit", "velocity_limit")):
            values[key][actuator.joint_indices] = getattr(actuator, attr)[0].detach().cpu()
    return {
        "delay_mode": delay_mode, "physics_dt": cfg.sim.dt, "decimation": cfg.decimation,
        "self_collisions": cfg.scene.robot.spawn.articulation_props.enabled_self_collisions,
        "asset": cfg.scene.robot.spawn.usd_path,
        **{key: value[joint_ids.cpu().long()] for key, value in values.items()},
    }


def validate_recording(data, joint_order, selected, current):
    if (data.get("experiment", {}).get("actual_clearance") or {}).get("passed") is False:
        raise ValueError("Recording failed leg clearance checks; redesign and recollect before fitting")
    if data.get("joint_names", joint_order) != list(joint_order):
        raise ValueError("Recording joint order differs from environment")
    for key in ("dof_pos", "des_dof_pos"):
        if data[key].shape != (len(data["time"]), len(joint_order)) or not torch.isfinite(data[key]).all():
            raise ValueError(f"Invalid shape or nonfinite values in {key}")
    time = data["time"].cpu().double()
    dt = current["physics_dt"] * current["decimation"]
    if len(time) < 2 or not torch.allclose(time[1:] - time[:-1], torch.full_like(time[1:], dt), atol=2e-6, rtol=1e-4):
        raise ValueError("Recording sampling interval differs from environment step_dt")
    if "excited" in data and not {joint_order[j] for j in selected} <= set(data["excited"]):
        raise ValueError("Requested fit includes unexcited joints; select the matching --group")
    recorded = data.get("experiment", {})
    for key in ("delay_mode", "physics_dt", "decimation", "self_collisions", "asset"):
        if key in recorded and recorded[key] != current[key]:
            raise ValueError(f"Recording/environment mismatch: {key}")
    for key in ("kp", "kd", "effort_limit", "velocity_limit"):
        if key in recorded and not torch.allclose(recorded[key].cpu(), current[key].cpu()):
            raise ValueError(f"Recording/environment mismatch: {key}")
