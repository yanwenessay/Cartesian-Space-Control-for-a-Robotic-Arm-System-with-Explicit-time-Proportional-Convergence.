注意：没有关节限幅的使用是及其危险的，请大家一定编写代码时加上安全保护
Note: The use without joint limiters is extremely dangerous. Please make sure to add safety protection when writing the code


# Cartesian-Space-Control-for-a-Robotic-Arm-System-with-Explicit-time-Proportional-Convergence.
[1]Cartesian space control and joint tracking control for a robotic arm system with explicit-time proportional convergence, IEEE/CAA J. Autom. Sinica, early access, 2026.  doi:  10.1109/JAS.2026.125963
[2]Trajectory planning and low-chattering fixed-time nonsingular terminal sliding mode control for a dual-arm free-floating space robot[J]. Robotica, 2022, 40(3): 625-645.

全新的位姿级间接笛卡尔空间控制模型，可将李雅普诺夫控制框架引入关节型机械臂的运动学闭环规划中，解决了速度级规划的解析解问题，有望为大范围位置规划提供最后1cm的速度解。
缺陷：解的不稳定性，在大范围规划中容易产生抖震，有待采用优化控制的方法对控制输入进行约束，使得轨迹平滑。A brand-new indirect Cartesian spatial control model at the pose level can introduce the Lyapunov control framework into the kinematic closed-loop planning of articulated robotic arms, solving the analytical solution problem of velocity level planning and is expected to provide the last 1cm velocity solution for large-scale position planning. Defect: The instability of the solution makes it prone to chattering in large-scale planning. It is necessary to adopt an optimized control method to constrain the control input to make the trajectory smooth.


为探索现有机械臂运动控制方法收敛速度较慢的本质原因，提出了位姿级间接笛卡尔空间控制模型（IEEE/CAA JAS, 2026. doi: 10.1109/JAS.2026.125963），利用机械臂末端位姿误差状态反馈和关节角速度跟踪误差假设（机械臂末端空间速度跟踪误差‖V_e-V_ep ‖和末端位姿跟踪误差‖ϵ_e ‖的有界性假设在真实物理环境中可被证实成立），解析机械臂末端的速度级位姿误差在关节角速度控制下的响应规律，构建位姿级笛卡尔空间间接控制系统模型：
	ϵ ̇_e=J_m θ ̇_p+∆V-V_ed ，
式中，ϵ ̇_e是机械臂末端的速度级位姿误差，∆V=V_e-V_ep是末端空间速度跟踪误差，V_ep是规划的机械臂末端空间速度，V_ed是期望的机械臂末端空间速度，θ ̇_p是被规划的关节角速度（即控制输入）。To explore the essential reasons for the slow convergence speed of the existing motion control methods for robotic arms, a pose level indirect Cartesian spatial control model is proposed (IEEE/CAA JAS, 2026. doi: (10.1109/JAS.2026.125963), using the state feedback of the end pose error of the robotic arm and the assumption of joint angular velocity tracking error (the boundedness assumption of the end spatial velocity tracking error of the robotic arm ‖V_e-V_ep ‖ and the end pose tracking error ‖ϵ_e ‖ can be verified in the real physical environment), Analyze the response law of the velocity level pose error at the end of the robotic arm under the control of joint angular velocity, and construct a pose level Cartesian space indirect control system model: ϵ ̇_e=J_m θ ̇_p+∆V-V_ed, where ϵ ̇_e is the velocity level pose error at the end of the robotic arm, ∆V=V_e-V_ep is the spatial velocity tracking error at the end, V_ep is the planned spatial velocity at the end of the robotic arm, and V_ed is the expected spatial velocity at the end of the robotic arm. θ ̇_p is the planned angular velocity of the joint (i.e., the control input).


