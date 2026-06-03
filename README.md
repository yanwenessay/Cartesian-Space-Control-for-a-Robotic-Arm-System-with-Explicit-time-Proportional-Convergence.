2026.6.3更新：（files：yanwen_exp_kinematic）
上一版论文（10.1109/JAS.2026.125963）程序采用小姿态误差近似，直接将几何角速度误差作为轴角姿态误差导数，并用几何雅可比进行位姿反馈规划。该处理在短距离运动中可近似成立，但在座舱长距离姿态变化任务中不够严谨。本代码包修正为基于 SO(3) 逆左雅可比和 analytical Jacobian 的位姿反馈规划器，并输出规划关节轨迹用于后续跟踪控制。（新论文：YAN W, ZHU R H, QIU Y H, PAN H Y, WU E Q. Adaptive T-S fuzzy impedance control for aircraft cockpit human-robot interaction system with dynamic uncertainties[Z]. IEEE THMS。Manuscript under review, 2026.）本仓库通过引入基于 J_l^{-1}(phi_e) 和 J_a 的 analytical pose-error planner，修正了此前小姿态误差近似实现中基于几何雅可比的位姿反馈规划问题。修正后的代码适用于大范围笛卡尔空间位姿规划，并生成规划关节轨迹，用于后续关节跟踪控制。
The previous code used a small-attitude-error approximation, where the geometric angular-velocity error was directly treated as the derivative of the axis-angle attitude error. This is acceptable for short-range motion but is not rigorous for long-range cockpit operation. This package provides the corrected analytical pose-feedback planner using the SO(3) inverse left Jacobian and analytical Jacobian, and generates the planned joint trajectory for downstream tracking control.（New paper：YAN W, ZHU R H, QIU Y H, PAN H Y, WU E Q. Adaptive T-S fuzzy impedance control for aircraft cockpit human-robot interaction system with dynamic uncertainties[Z]. IEEE THMS。Manuscript under review, 2026.）This repository corrects the previous small-attitude-error implementation by replacing the geometric-Jacobian-based pose feedback with an analytical pose-error planner using J_l^{-1}(phi_e) and J_a. The corrected code is suitable for large-range Cartesian pose planning and generates planned joint trajectories for subsequent joint tracking control.

<img width="1706" height="1279" alt="experiment_photo_20260603" src="https://github.com/user-attachments/assets/2c9cea06-3b0f-45a5-872c-b0de943dd406" />
https://github.com/user-attachments/assets/e941b028-dba6-4f91-a77f-c28ba0319ae9 (-2s to 10s, demo)

2026.5.1更新：
[1]Cartesian space control and joint tracking control for a robotic arm system with explicit-time proportional convergence, IEEE/CAA J. Autom. Sinica, early access, 2026.  doi:  10.1109/JAS.2026.125963（files：code_of_paper[1]、python_Robot_arm_CSIC_kinematic_planning）
注意：没有关节限幅的使用是及其危险的，请编写代码时加上安全保护
Note: The use without joint limiters is extremely dangerous. Please make sure to add safety protection when writing the code

2026.2月最初版：（files：code_of_paper[1]、python_Robot_arm_CSIC_kinematic_planning）
# Cartesian-Space-Control-for-a-Robotic-Arm-System-with-Explicit-time-Proportional-Convergence. （ϵ̇_e = J_a θ̇_p + ΔV_a − V_ed^a）
[1]Cartesian space control and joint tracking control for a robotic arm system with explicit-time proportional convergence, IEEE/CAA J. Autom. Sinica, early access, 2026.  doi:  10.1109/JAS.2026.125963（files：code_of_paper[1]、python_Robot_arm_CSIC_kinematic_planning）
[2]Trajectory planning and low-chattering fixed-time nonsingular terminal sliding mode control for a dual-arm free-floating space robot[J]. Robotica, 2022, 40(3): 625-645.
实现位姿级间接笛卡尔空间控制模型，用于关节型机械臂的速度级闭环规划。该方法根据末端位姿误差在线生成规划关节角速度 θ̇_p，而不是直接预设关节轨迹。末端误差定义为 ϵ_e = [e_e, φ_e]，其中 e_e = p_e − p_ed，R_φe = R_e R_ed^T，φ_e 为轴角姿态误差。由于轴角误差导数不能直接等同于几何角速度误差，本版本采用 SO(3) 逆左雅可比 J_l^-1(φ_e)，并构造 analytical Jacobian：J_a = diag(I, J_l^-1(φ_e)) J_m。对应的位姿级控制模型为：ϵ̇_e = J_a θ̇_p + ΔV_a − V_ed^a，其中 ΔV_a = J_a(θ̇ − θ̇_p)。代码输出规划关节轨迹 θ_p、θ̇_p、θ̈_p，已经用于关节跟踪控制、Isaac 仿真和 Kinova Gen3 、realman rm65b/75b、大然6轴标准型 实机实验。
This version implements a pose-level indirect Cartesian-space control model for velocity-level closed-loop planning of articulated robotic arms. The planner generates the planned joint angular velocity θ̇_p online from the end-effector pose error, instead of prescribing a joint trajectory directly. The pose error is defined as ϵ_e = [e_e, φ_e], where e_e = p_e − p_ed, R_φe = R_e R_ed^T, and φ_e is the axis-angle attitude error. Since the derivative of the axis-angle error is not generally equal to the geometric angular-velocity error, this version uses the SO(3) inverse left Jacobian J_l^-1(φ_e) and constructs the analytical Jacobian J_a = diag(I, J_l^-1(φ_e)) J_m. The resulting pose-level control model is ϵ̇_e = J_a θ̇_p + ΔV_a − V_ed^a, where ΔV_a = J_a(θ̇ − θ̇_p). The code outputs θ_p, θ̇_p, and θ̈_p for downstream joint tracking control, Isaac simulation, and Kinova Gen3 real-robot experiments.

## License

This project is licensed under the Apache License 2.0.
