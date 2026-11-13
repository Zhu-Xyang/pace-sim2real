"""Offline comparison on identical command/time grids; no time shifting."""
import argparse,csv
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--real',type=Path,required=True);p.add_argument('--sim',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();r=torch.load(a.real,weights_only=True,map_location='cpu');s=torch.load(a.sim,weights_only=True,map_location='cpu')
assert r['joint_names']==s['joint_names'],'Joint order mismatch'
assert r['des_dof_pos'].shape==s['des_dof_pos'].shape
assert torch.allclose(r['des_dof_pos'],s['des_dof_pos'],atol=1e-7,rtol=0),'Targets differ'
assert torch.allclose(r['time'].double(),s['time'].double(),atol=2e-6,rtol=0),'Times differ'
for k in ('kp','kd'):assert torch.allclose(r['experiment'][k],s['experiment'][k]),k+' differs'
a.output.mkdir(parents=True,exist_ok=True);names=r['joint_names'];ids=[names.index(n) for n in r['excited']];t=r['time'].numpy();metrics=[]
for key,unit in [('dof_pos','rad'),('dof_vel','rad/s'),('dof_torque','Nm')]:
 fig,axes=plt.subplots(len(ids),1,figsize=(13,3*len(ids)),squeeze=False)
 for ax,i in zip(axes.flat,ids):
  ax.plot(t,r[key][:,i],label='Real feedback');ax.plot(t,s[key][:,i],label='Sim feedback',alpha=.8)
  if key=='dof_pos':
   ax.plot(t,r['des_dof_pos'][:,i],'--',label='Recorded command',alpha=.7)
   delta=s[key][:,i]-r[key][:,i];metrics.append(dict(joint=names[i],rmse_rad=float(delta.square().mean().sqrt()),max_error_rad=float(delta.abs().max())))
  ax.set(title=names[i],xlabel='Phase 2 time [s]',ylabel=unit);ax.legend();ax.grid(alpha=.3)
 fig.tight_layout();fig.savefig(a.output/(key+'.png'),dpi=160);plt.close(fig)
with (a.output/'metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=metrics[0]);w.writeheader();w.writerows(metrics)
(a.output/'report.md').write_text('# 真机与仿真记录对照\n\n目标、时间网格和PD已校验一致；未平移对齐、未拟合bias。仿真使用采集默认动力学，非本次fit参数。力矩为真机SDK反馈与仿真记录值，两者测量/采样口径不同，不能直接等同。\n\n'+''.join(f'![{k}]({k}.png)\n\n' for k in ('dof_pos','dof_vel','dof_torque'))+'[位置误差](metrics.csv)\n')
print(a.output/'report.md')
