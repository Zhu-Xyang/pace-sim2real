"""Export complete SDK phase-2 data to 26-DoF PACE without scaling or fabricated GT."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from wrist_diagnostic import ROOT, WRISTS


def export(directory, output):
    sdk = ROOT.parent/'engineai_robotics_native_sdk/tools/sys_id/export_pace.py'
    spec = importlib.util.spec_from_file_location('sdk_pace_export', sdk)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    manifest, arrays, quality = module.load_recording(directory)
    if manifest['group'] != 'arms' or manifest['runtime_simulation']:
        raise ValueError('Expected a real arms recording')
    with (directory/'raw.csv').open() as f:
        header=f.readline().strip().split(','); raw=np.loadtxt(f,delimiter=',')
    rows=raw[raw[:,header.index('phase')]==2]
    names=sorted(manifest['joint_names']+list(WRISTS))
    if len(names)!=26 or len(set(names))!=26:
        raise ValueError('Expected 22 original joints plus four wrists')
    def columns(key):
        return rows[:,[header.index(key+'/'+n) for n in names]]
    kp,kd=columns('kp'),columns('kd')
    if not np.all(kp==kp[0]) or not np.all(kd==kd[0]):
        raise ValueError('Phase-2 gains vary; constant-gain replay would be invalid')
    for key,full in [('kp','full_kp'),('kd','full_kd')]:
        expected=np.array([manifest[full][manifest['full_joint_names'].index(n)] for n in names])
        if not np.allclose(columns(key),expected,atol=1e-8,rtol=1e-7):
            raise ValueError('Manifest/raw gain mismatch')
    for field,key in [('dof_pos','q'),('des_dof_pos','q_des'),('dof_vel','qd'),('dof_torque','tau_feedback')]:
        a=columns(key)
        if not np.isfinite(a).all():raise ValueError('Nonfinite joint data')
        if not np.array_equal(a[:,[names.index(n) for n in manifest['joint_names']]],arrays[field]):
            raise ValueError('26-DoF export differs from validated 22-DoF data')
        arrays[field]=a
    experiment=dict(physics_dt=float(manifest['sample_dt']),decimation=1,delay_mode='command',
                    group='arms',duration=manifest['duration'],rest=manifest['rest'],ramp=manifest['ramp'],
                    f0=manifest['f0'],kp=torch.tensor(kp[0],dtype=torch.float32),kd=torch.tensor(kd[0],dtype=torch.float32),
                    source='real SDK phase 2; wrist hold, no amplitude rescaling',runtime_simulation=False)
    for key in ('center_rad','signed_amplitude_rad','f1_per_joint'):
        values=[manifest[key][manifest['joint_names'].index(n)] if n in manifest['joint_names'] else
                (manifest['f0'] if key=='f1_per_joint' else manifest['hold_position'][manifest['full_joint_names'].index(n)] if key=='center_rad' else 0.) for n in names]
        experiment[key]=torch.tensor(values,dtype=torch.float32)
    quality.update(export_profile='wrist_hold_26dof',sdk_exporter_sha256=hashlib.sha256(sdk.read_bytes()).hexdigest(),
                   source_sha256={n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in ('raw.csv','experiment.yaml','status.yaml','trajectory.csv')},
                   limitations='Wrist motion preserved. Wrist dynamics unknown; local model limits are not verified SDK runtime limits. Torso/head/fingers remain fixed in simulation.')
    data={k:torch.tensor(v,dtype=torch.float64 if 'time' in k else torch.float32) for k,v in arrays.items()}
    data.update(joint_names=names,excited=manifest['excited'],experiment=experiment,capture_manifest=manifest,quality=quality)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f:torch.save(data,f)
    output.with_suffix('.quality.json').write_text(json.dumps(quality,indent=2)+'\n')
    print(f'Saved {output}: {len(data["time"])} x {len(names)}; 10 excited joints; wrist gains from raw; no GT')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();export(args.recording,args.output)
