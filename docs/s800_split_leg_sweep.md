# 腿部分组扫频（2026-11-06）

`data_collection.py --group legs --trajectory_profile legacy` 新增 `--leg_sweep`，默认 all 保持原行为。hips 与 knee_ankle 是两次独立采集，不是同一条轨迹自动切换。两侧同时激励，整机状态仍记录，excited 仅含六个实际激励关节。

- hips：髋部放大候选：pitch 中心 -.55、幅值 -.30；左 roll 中心 .24、幅值 .18（右侧镜像）；yaw 幅值左右 ±.10。按用户要求在上一版 .25/.15/.04 的基础上继续扩大，供仿真观察，roll 中心同步外移，保留 .06 rad 最小向外目标。膝踝保持当前中心；手臂外张保持。仅此模式使用这组候选，不改 nominal_motion 或 knee_ankle。已检查目标方向及加 .05 rad 偏置后的 URDF 角度限位；尚未动态验证支架间隙，不是安全最大幅值。
- knee_ankle：髋 pitch 固定为 `--hip_pitch_hold`（默认 -.8 rad，待验证的抬腿姿态）；髋 roll 保持左右 ±.16、yaw 为 0。膝中心 .4184、幅值 .4；踝 pitch 中心 0、幅值 .2176；踝 roll 中心左右 ∓.007308、幅值 ±.084 rad。恢复的是十月实际采集的原始 scale×0.8，而非未经缩放的名义幅值。

参数在 `scripts/pace/s800_motion.py::split_leg_motion`。模式不能与 trajectory_plan、额外 scale_factor、clearance profile 混用，避免静默覆盖。默认输出名带模式后缀，也可显式指定 output。

仿真直接初始化到保持姿态，未设计真机入场抬腿路径。支架方向检查不等于无碰撞证明；请保留支架接触与自碰撞检查。只改变这些轨迹参数，不修改原 PD/GT/延迟。

目前 fit 的 legs 分组仍包含全部腿部关节：不能直接对这些部分激励数据运行 `fit --group legs`；现有 excited 校验会拒绝，后续拟合需匹配自定义关节集合。

离线验证了两种模式的六关节选择、其余关节恒定目标和支架方向检查；未运行动态仿真。
