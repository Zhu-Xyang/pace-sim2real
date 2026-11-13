"""Export real split-leg captures; aborted prefixes require explicit diagnostic opt-in.

No resampling, synthetic tail, GT, or source-file modification. The 22-DoF replay
holds torso/wrists mechanically; measured compliance remains a model limitation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from experiment_utils import select_leg_sweep


def export(directory, output, *, allow_partial=False):
    directory, output = Path(directory), Path(output)
    manifest = yaml.safe_load((directory / 'experiment.yaml').read_text())
    status = yaml.safe_load((directory / 'status.yaml').read_text())
    partial = status['result'] != 'complete'
    if partial and (not allow_partial or not str(status['result']).startswith('aborted:')):
        raise ValueError('Aborted data requires --allow_partial; unknown statuses are rejected')
    if manifest['runtime_simulation'] or manifest['group'] != 'legs':
        raise ValueError('Expected real split-leg capture')
    names = manifest['joint_names']
    if len(names) != 22 or len(set(names)) != 22:
        raise ValueError('Expected unique 22-DoF joint names')
    mode = (manifest.get('joint_type') or manifest.get('leg_sweep', '')).lower()
    if manifest.get('single_joint'):
        raise ValueError('This exporter expects a bilateral pair, not a one-sided joint')
    selected = select_leg_sweep(names, mode)
    excited = [names[i] for i in selected]
    if set(manifest['excited']) != set(excited):
        raise ValueError('Excited joints do not match split-leg group')
    with (directory / 'raw.csv').open() as f:
        header = next(csv.reader(f))
        raw = np.loadtxt(f, delimiter=',', ndmin=2)
    if len(set(header)) != len(header) or raw.shape != (status['rows'], len(header)) or not np.isfinite(raw).all():
        raise ValueError('Malformed/nonfinite recording or status row mismatch')
    get = lambda key: raw[:, header.index(key)]
    if not np.array_equal(get('tick'), np.arange(len(raw))):
        raise ValueError('Missing raw ticks')
    phase = get('phase')
    if not np.isin(phase, [0, 1, 2]).all() or np.any(np.diff(phase) < 0):
        raise ValueError('Unexpected phase sequence')
    rows = raw[phase == 2]
    count = len(rows); planned_count = manifest['sample_count']
    if count < 2 or count > planned_count or (not partial and count != planned_count):
        raise ValueError('Invalid phase-2 length')
    if not np.array_equal(rows[:, header.index('sample')], np.arange(count)):
        raise ValueError('Phase-2 data must be a contiguous prefix starting at sample zero')
    dt = float(manifest['sample_dt'])
    if dt != .005:
        raise ValueError('Expected 200 Hz recording')
    stamps = rows[:, header.index('monotonic_s')]
    time = np.arange(count) * dt
    jitter = stamps - stamps[0] - time
    interval_jitter = np.diff(stamps) - dt
    if np.any(np.diff(stamps) <= 0) or max(abs(jitter).max(), abs(interval_jitter).max()) > .001:
        raise ValueError('Timing jitter exceeds 1 ms; refusing to resample')
    def columns(key, joints=names):
        return rows[:, [header.index(key + '/' + n) for n in joints]]
    with (directory / 'trajectory.csv').open() as f:
        ph = next(csv.reader(f)); plan = np.loadtxt(f, delimiter=',', ndmin=2)
    if ph != ['time'] + names or plan.shape != (planned_count, len(names)+1) or not np.isfinite(plan).all():
        raise ValueError('Invalid planned trajectory layout')
    if not np.allclose(plan[:, 0], np.arange(planned_count)*dt, atol=1e-8, rtol=0):
        raise ValueError('Planned trajectory time grid mismatch')
    if not np.allclose(columns('q_des'), plan[:count, 1:], atol=1e-10, rtol=0):
        raise ValueError('Recorded commands differ from trajectory prefix')
    for key in ('kp', 'kd'):
        if not np.allclose(columns(key), np.array(manifest[key])[None, :], atol=1e-10, rtol=0):
            raise ValueError('Recorded gain mismatch')
    held = [n for n in manifest['full_joint_names'] if 'WRIST' in n or 'TORSO' in n]
    held_error = abs(columns('q', held)-columns('q_des', held)).max(axis=0)
    if held_error.max() > manifest['fixed_joint_error_limit']:
        raise ValueError('Fixed-joint approximation exceeds recorded tolerance')
    quality = dict(partial_diagnostic=partial, samples=count, planned_samples=planned_count,
                   missing_tail_samples=planned_count-count, export_profile='real_split_leg_22dof',
                   max_phase_jitter_s=float(abs(jitter).max()),
                   max_interval_jitter_s=float(abs(interval_jitter).max()),
                   held_joint_max_error_rad=dict(zip(held, map(float, held_error))),
                   limitations='Contact not ruled out. Torso/wrists fixed in replay, unlike real PD hold. '
                               'No trigger frame, no hardware sensor timestamps. Frozen dynamics are priors.',
                   source_sha256={n:hashlib.sha256((directory/n).read_bytes()).hexdigest()
                                  for n in ('raw.csv','trajectory.csv','experiment.yaml','status.yaml')})
    experiment = dict(physics_dt=dt, decimation=1, delay_mode='command', group='legs', leg_sweep=mode,
                      runtime_simulation=False, duration=manifest['duration'], rest=manifest['rest'],
                      ramp=manifest['ramp'], f0=manifest['f0'], source='SDK split-leg capture; actual prefix only')
    # Hardware limits remain in capture_manifest; not asserted equal to simulator limits.
    for key in ('kp','kd','center_rad','signed_amplitude_rad','f1_per_joint'):
        experiment[key] = torch.tensor(manifest[key], dtype=torch.float32)
    arrays = dict(time=time, time_actual=stamps-stamps[0], monotonic_time=stamps,
                  dof_pos=columns('q'), des_dof_pos=columns('q_des'),
                  dof_vel=columns('qd'), dof_torque=columns('tau_feedback'))
    data = {k:torch.tensor(v, dtype=torch.float64 if 'time' in k else torch.float32) for k,v in arrays.items()}
    data.update(joint_names=names, excited=excited, experiment=experiment,
                capture_manifest=manifest, capture_status=status, quality=quality)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as f:
        torch.save(data, f)
    output.with_suffix('.quality.json').write_text(json.dumps(quality, indent=2)+'\n')
    print(f'Saved {output}: {count} x 22, {mode}, partial_diagnostic={partial}; no GT or synthetic tail')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow_partial', action='store_true', help='Explicit diagnostic export of aborted prefix')
    args = parser.parse_args()
    export(args.recording, args.output, allow_partial=args.allow_partial)
