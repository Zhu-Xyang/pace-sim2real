"""Append reproducible bilateral checks to existing S800 reports (no simulation)."""
from pathlib import Path
import csv
import torch
ROOT=Path(__file__).resolve().parents[3]
MARK='## 左右一致性检查'
MIRRORED={'HIP_ROLL','HIP_YAW','ANKLE_ROLL','SHOULDER_ROLL','SHOULDER_YAW','ELBOW_YAW','WRIST_ROLL'}

def read(path):
    with path.open() as f:return list(csv.DictReader(f))

def write(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def append(out,lines):
    p=out/'report.md';text=p.read_text()
    if MARK in text:text=text[:text.index(MARK)]
    p.write_text(text.rstrip()+'\n\n'+'\n'.join(lines)+'\n')

def pairs(rows):
    d={(r['joint'][4:-2],r['joint'][-1]):r for r in rows if r['joint'].endswith(('_L','_R'))}
    return [(k,d[k,'L'],d[k,'R']) for k,side in d if side=='L' and (k,'R') in d]

def fit_report(folder,filename,columns,filter_optimized=False):
    out=ROOT/'docs/reports/s800_real'/folder
    rows=read(out/filename)
    if filter_optimized:rows=[r for r in rows if r['optimized']=='True']
    cfg=torch.load(out/'snapshot_config.pt',map_location='cpu',weights_only=True)
    order=cfg['joint_order'];n=len(order);bounds=cfg['bounds']
    def edge(name,block,value):
        lo,hi=map(float,bounds[block*n+order.index(name)])
        return min(value-lo,hi-value)/(hi-lo)<=.01
    result=[];bias=[]
    lines=[MARK,'','使用本报告的**历史最佳代**参数，仅比较本次实际优化的左右配对关节；冻结参数不作为辨识一致性的证据。','',r'动力学参数的对称相对差定义为 $D=2|L-R|/(|L|+|R|)\times100\%$。该指标不依赖选择哪一侧作分母；不是参数误差，也不是统计置信度。','', '当两侧绝对值都小于 1e-4（各参数自身单位）时，不给相对百分比，避免近零放大。该阈值只是数值展示规则，不是物理合格标准。','', '|关节类型|参数|左侧|右侧|绝对差|对称相对差|边界提示|','|---|---|---:|---:|---:|---:|---|']
    for kind,l,r in pairs(rows):
        for block,key in enumerate(('armature','viscous','friction')):
            a,b=float(l[columns[key]]),float(r[columns[key]])
            rel=200*abs(a-b)/(abs(a)+abs(b)) if max(abs(a),abs(b))>=1e-4 else None
            near=[side for side,row,v in [('L',l,a),('R',r,b)] if edge(row['joint'],block,v)]
            note=','.join(near)+'距边界≤1%' if near else '—'
            result.append(dict(joint_type=kind,parameter=key,left=a,right=b,absolute_difference=abs(a-b),symmetric_difference_pct=rel,near_bound=','.join(near)))
            lines.append(f'|{kind}|{key}|{a:.8f}|{b:.8f}|{abs(a-b):.8f}|'+(f'{rel:.2f}%' if rel is not None else '近零，不计算')+f'|{note}|')
        a,b=float(l[columns['bias']]),float(r[columns['bias']]);sign=-1 if kind in MIRRORED else 1
        bias.append(dict(joint_type=kind,left_bias_rad=a,right_bias_rad=b,raw_difference_rad=abs(a-b),mirror_sign=sign,right_mirror_aligned_rad=sign*b,mirror_difference_rad=abs(a-sign*b)))
    lines+=['','### Bias：原始符号与镜像对齐','', '动力学参数无需镜像变号；bias需区分坐标方向。以下按 s800_motion 的轨迹镜像约定，roll/yaw部分轴将右侧取反后比较。**这是坐标约定下的诊断，不表示独立编码器零位误差必然左右相等或镜像对称。** 不用bias百分比避免零附近或异号误导。','', '|关节类型|左bias rad|右bias rad|原始绝对差 rad|右侧镜像系数|镜像后绝对差 rad|','|---|---:|---:|---:|---:|---:|']
    for r in bias:lines.append(f'|{r["joint_type"]}|{r["left_bias_rad"]:.6f}|{r["right_bias_rad"]:.6f}|{r["raw_difference_rad"]:.6f}|{r["mirror_sign"]:+d}|{r["mirror_difference_rad"]:.6f}|')
    largest=sorted([r for r in result if r['symmetric_difference_pct'] is not None],key=lambda r:r['symmetric_difference_pct'],reverse=True)[:3]
    lines+=['','### 结果解读','']
    for r in largest:lines.append(f'- 差异较大的项目：{r["joint_type"]} {r["parameter"]}，对称相对差 {r["symmetric_difference_pct"]:.2f}%，绝对差 {r["absolute_difference"]:.6g}。')
    lines+=['- 左右同时贴到同一边界而数值接近，可能是共同的约束/模型补偿，不能当成参数测准的证据。','- 左右硬件可能存在真实摩擦、线缆、装配和标定差异；也可能来自激励不足、冻结参数、安装条件或优化参数互相补偿。没有重复实验与不确定度，不能将本表直接定为合格/不合格。','', '[动力学左右对比 CSV](bilateral_parameters.csv) · [bias 对比 CSV](bilateral_bias.csv)','']
    write(out/'bilateral_parameters.csv',result);write(out/'bilateral_bias.csv',bias);append(out,lines)
    print(folder,[(r['joint_type'],r['parameter'],round(r['symmetric_difference_pct'],2)) for r in largest])

fit_report('hips_fit_partial_26_11_10_11-17-55','parameters_comparison.csv',{k:k+'_best' for k in ('armature','viscous','friction','bias')})
fit_report('arm_fit_scale25_26_11_09_20-35-59','best_vs_saved_parameters.csv',{k:k for k in ('armature','viscous','friction','bias')})
fit_report('arm_fit_scale20_26_11_06_17-58-54','best_vs_final_parameters.csv',dict(armature='best_armature',viscous='best_viscous_friction',friction='best_coulomb_friction',bias='best_bias'),True)
out=ROOT/'docs/reports/s800_natural_frequencies_20261110';rows=read(out/'joint_frequencies.csv');checks=[]
lines=[MARK,'','比较左右同类型关节的输入Kp/Kd、XML armature/粘滞与计算结果。这是**配置一致性检查**，不是实测硬件对称性验证。','', '|关节类型|左fn Hz|右fn Hz|fn绝对差 Hz|Kp/Kd/armature/XML阻尼一致|','|---|---:|---:|---:|---|']
for kind,l,r in pairs(rows):
    same=all(float(l[k])==float(r[k]) for k in ('kp','kd','xml_armature_kg_m2','xml_viscous'))
    diff=abs(float(l['fn_hz'])-float(r['fn_hz']))
    checks.append(dict(joint_type=kind,left_fn_hz=float(l['fn_hz']),right_fn_hz=float(r['fn_hz']),difference_hz=diff,inputs_equal=same))
    lines.append(f'|{kind}|{float(l["fn_hz"]):.6f}|{float(r["fn_hz"]):.6f}|{diff:.6g}|'+('是' if same else '否')+'|')
lines+=['',f'共检查 {len(checks)} 对关节。躯干和两个头部关节无左右配对，不参与此检查。相同的输入自然会产生相同的armature-only频率，不能据此证明实际左右闭环动态一致。','', '[左右配置检查 CSV](bilateral_frequencies.csv)']
write(out/'bilateral_frequencies.csv',checks);append(out,lines)
