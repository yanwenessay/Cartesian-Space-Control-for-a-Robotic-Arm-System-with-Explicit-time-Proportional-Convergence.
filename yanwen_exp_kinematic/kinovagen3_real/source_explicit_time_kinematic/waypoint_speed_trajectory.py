import os
import sys
import time
import math
import numpy as np

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2

# -----------------------------------------------------------------------------
# 集中参数
# -----------------------------------------------------------------------------
FREQUENCY_HZ = 2000.0
DT = 1.0 / FREQUENCY_HZ

Z_TOOL = -0.16746
POS_TOLERANCE_M = 0.0001
ANGLE_TOLERANCE_RAD = 0.1 * math.pi / 180.0

MAX_JOINT_SPEED_DEG_S = 30.0
MAX_JOINT_SPEED_RAD_S = math.radians(MAX_JOINT_SPEED_DEG_S)
SPEED_SCALE = 0.8

SEGMENT_DURATION_S = 5.0  # 每段5秒
MAX_TOTAL_SECONDS = 30.0

# 初始关节角（度）来自 YWtorque_test.py (1143-1144)
INITIAL_JOINT_ANGLES_DEG = [10, -20, 10, 120, 60, 10, 70]

# 轨迹 waypoints（末端位置 P 单位 m，姿态 Phi_deg 单位 度）
WAYPOINTS = [
    {"P": np.array([0.416, -0.039, 0.517]), "Phi_deg": np.array([-134.3, -33.6, -128.3])},
    {"P": np.array([0.375, 0.020, 0.237]), "Phi_deg": np.array([-91.5, -49.4, -160.1])},
    {"P": np.array([0.415, -0.036, 0.143]), "Phi_deg": np.array([-97.3, -63.9, -143.3])},
    {"P": np.array([0.432, -0.055, 0.146]), "Phi_deg": np.array([-88.3, -66.2, -151.4])},
    {"P": np.array([0.440, -0.015, 0.124]), "Phi_deg": np.array([-97.8, -79.5, -133.7])},
    {"P": np.array([0.440, -0.015, 0.124]), "Phi_deg": np.array([-97.8, -79.5, -133.7])},
]


def planning_wrapper(P_target, Phi_target, theta):
    # 运行时导入实际规划函数
    from planning_main import planning
    return planning(P_target, Phi_target, theta)


def Kinematic(theta, z_tool):
    # 运行时导入实际运动学函数
    from Kinematic_fcn import Kinematic as KinematicImpl
    return KinematicImpl(theta, z_tool)


class WaypointSpeedTrajectory:
    def __init__(self, base: BaseClient, base_cyclic: BaseCyclicClient):
        self.base = base
        self.base_cyclic = base_cyclic

        # Servo 模式
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        self.actuator_count = self.base.GetActuatorCount().count
        if self.actuator_count != 7:
            raise ValueError(f"需要7轴机械臂，当前为 {self.actuator_count} 轴")

        self.iteration = 0
        self.start_time = time.time()

    def move_to_initial_joint_angles(self):
        constrained = Base_pb2.ConstrainedJointAngles()
        for j in range(self.actuator_count):
            ja = constrained.joint_angles.joint_angles.add()
            ja.joint_identifier = j
            # 如果提供的初始角度不足7个，缺失的补0
            value_deg = INITIAL_JOINT_ANGLES_DEG[j] if j < len(INITIAL_JOINT_ANGLES_DEG) else 0.0
            ja.value = value_deg
        print("Reaching initial joint angles...")
        self.base.PlayJointTrajectory(constrained)
        time.sleep(3.0)

    def get_actual_joint_angles(self):
        fb = self.base_cyclic.RefreshFeedback()
        angles_deg = [fb.actuators[i].position for i in range(self.actuator_count)]
        return np.deg2rad(np.array(angles_deg))

    def compute_pose(self, theta):
        P, Phi, _, _ = Kinematic(theta, Z_TOOL)
        return P, Phi

    def compute_joint_speed(self, theta, P_target, Phi_target):
        theta_dot = np.array(planning_wrapper(P_target, np.deg2rad(Phi_target), theta))
        theta_dot = np.clip(theta_dot, -MAX_JOINT_SPEED_RAD_S, MAX_JOINT_SPEED_RAD_S)
        theta_dot *= SPEED_SCALE
        return theta_dot

    def send_joint_speeds(self, theta_dot_rad_s):
        js = Base_pb2.JointSpeeds()
        for j in range(self.actuator_count):
            sp = js.joint_speeds.add()
            sp.joint_identifier = j
            sp.value = math.degrees(theta_dot_rad_s[j])
            sp.duration = 0
        self.base.SendJointSpeedsCommand(js)

    def run_waypoints(self):
        total_start = time.time()
        # Histories
        time_history = []              # seconds from start
        theta_dot_hist = []            # (rad/s)
        err6_hist = []                 # [dx,dy,dz, dphi_x,dphi_y,dphi_z]
        stop_all = False
        try:
            for idx, wp in enumerate(WAYPOINTS):
                if stop_all:
                    break
                P_target = wp["P"]
                Phi_target_deg = wp["Phi_deg"]
                print(f"\n=== 段 {idx+1}/{len(WAYPOINTS)} | 目标P={P_target}, 目标Phi(deg)={Phi_target_deg} | 持续 {SEGMENT_DURATION_S}s ===")
                seg_start = time.time()
                while time.time() - seg_start < SEGMENT_DURATION_S:
                    loop_t0 = time.perf_counter()

                    theta = self.get_actual_joint_angles()
                    # 可选：估计当前误差，仅供打印
                    try:
                        P, Phi = self.compute_pose(theta)
                        pos_err = np.linalg.norm(P - P_target)
                        ang_err = np.linalg.norm(Phi - np.deg2rad(Phi_target_deg))
                        # record 6D error
                        err6 = np.concatenate([P - P_target, Phi - np.deg2rad(Phi_target_deg)])
                    except Exception:
                        pos_err, ang_err = 0.0, 0.0
                        err6 = np.zeros(6)

                    theta_dot = self.compute_joint_speed(theta, P_target, Phi_target_deg)
                    self.send_joint_speeds(theta_dot)

                    if self.iteration % int(0.5 / DT) == 0:
                        print(f"it={self.iteration:7d} | pos_err={pos_err*1000:7.3f} mm | ang_err={math.degrees(ang_err):7.3f} deg | |theta_dot|={np.linalg.norm(theta_dot):.3f} rad/s")

                    self.iteration += 1
                    # record histories
                    time_history.append(time.time() - total_start)
                    theta_dot_hist.append(theta_dot.copy())
                    err6_hist.append(err6.copy())
                    # 2000Hz 节拍
                    elapsed = time.perf_counter() - loop_t0
                    sleep_t = max(0.0, DT - elapsed)
                    if sleep_t > 0:
                        time.sleep(sleep_t)

                    if time.time() - total_start > MAX_TOTAL_SECONDS:
                        print("⏹️ 达到最大总时长，结束轨迹")
                        stop_all = True
                        break

            if not stop_all:
                print("\n✅ 轨迹执行完成")
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断，结束轨迹")
        finally:
            try:
                self.base.Stop()
            except Exception:
                pass

            # Save plots
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                base_dir = os.path.join(os.path.dirname(__file__), "trajectory_results")
                os.makedirs(base_dir, exist_ok=True)
                result_dir = os.path.join(base_dir, f"run-{timestamp}")
                os.makedirs(result_dir, exist_ok=True)

                t = np.array(time_history)
                if len(theta_dot_hist) > 0:
                    vel = np.array(theta_dot_hist)  # (N,7) rad/s
                    vel_deg = np.degrees(vel)       # 转为 deg/s
                    rows, cols = 4, 2
                    fig1, axes1 = plt.subplots(rows, cols, figsize=(14, 9), sharex=True)
                    fig1.suptitle("Manipulator Joint Velocity Curves")
                    ylabels = [
                        r"$\dot \Theta_1$ ($^o$/s)",
                        r"$\dot \Theta_2$ ($^o$/s)",
                        r"$\dot \Theta_3$ ($^o$/s)",
                        r"$\dot \Theta_4$ ($^o$/s)",
                        r"$\dot \Theta_5$ ($^o$/s)",
                        r"$\dot \Theta_6$ ($^o$/s)",
                        r"$\dot \Theta_7$ ($^o$/s)"
                    ]
                    for j in range(min(7, vel_deg.shape[1])):
                        r, c = divmod(j, cols)
                        ax = axes1[r, c]
                        ax.plot(t, vel_deg[:, j], label=f"Joint {j+1}")
                        ax.set_ylabel(ylabels[j])
                        ax.grid(True, alpha=0.3)
                        ax.legend(loc='upper right')
                        if r == rows - 1 or (r == rows - 2 and cols == 2 and j >= 4):
                            ax.set_xlabel("Time (s)")
                    # 隐藏第8子图（若存在）
                    if rows * cols > 7:
                        axes1[-1, -1].axis('off')
                    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
                    fig1.savefig(os.path.join(result_dir, "joint_velocity_curves.png"))
                    plt.close(fig1)
                else:
                    fig1, _ = plt.subplots(figsize=(8, 4))
                    fig1.suptitle("Manipulator Joint Velocity Curves (no data)")
                    fig1.savefig(os.path.join(result_dir, "joint_velocity_curves.png"))
                    plt.close(fig1)

                if len(err6_hist) > 0:
                    err6 = np.array(err6_hist)  # (N,6)
                    pos_mm = err6[:, 0:3] * 1000.0
                    ang_deg = np.degrees(err6[:, 3:6])
                    fig2, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
                    fig2.suptitle("Manipulator End-effector Tracking Error Curves")
                    pos_labels = [r"$P_x$ (mm)", r"$P_y$ (mm)", r"$P_z$ (mm)"]
                    ang_labels = [r"$\Phi_x$ ($^o$)", r"$\Phi_y$ ($^o$)", r"$\Phi_z$ ($^o$)"]
                    for i in range(3):
                        ax = axes[0, i]
                        ax.plot(t, pos_mm[:, i], 'b-')
                        ax.set_ylabel(pos_labels[i])
                        ax.grid(True, alpha=0.3)
                        # inset: move x,y a bit up at lower-right; z stays upper-right
                        if i in (0, 1):
                            ins = ax.inset_axes([0.55, 0.10, 0.4, 0.4])
                        else:
                            ins = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
                        ins.plot(t, pos_mm[:, i], 'b-')
                        ins.set_ylim(-0.2, 0.2)
                        ins.set_yticks(np.arange(-0.2, 0.201, 0.1))
                        ins.grid(True, alpha=0.3)
                    for i in range(3):
                        ax = axes[1, i]
                        ax.plot(t, ang_deg[:, i], 'g-')
                        ax.set_ylabel(ang_labels[i])
                        ax.set_xlabel("Time (s)")
                        ax.grid(True, alpha=0.3)
                        # set y-limits for Phi_x and Phi_y big plots
                        if i == 0:
                            ax.set_ylim(-60, 35)
                        if i == 1:
                            ax.set_ylim(-30, 20)
                        # inset: move Phi_x,Phi_y a bit up at lower-right; Phi_z at upper-right
                        if i in (0, 1):
                            ins = ax.inset_axes([0.55, 0.10, 0.4, 0.4])
                        else:
                            ins = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
                        ins.plot(t, ang_deg[:, i], 'g-')
                        ins.set_ylim(-0.2, 0.2)
                        ins.set_yticks(np.arange(-0.2, 0.201, 0.1))
                        ins.grid(True, alpha=0.3)
                    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
                    fig2.savefig(os.path.join(result_dir, "ee_6d_error_tracking.png"))
                    plt.close(fig2)
                else:
                    fig2, _ = plt.subplots(figsize=(8, 4))
                    fig2.suptitle("Manipulator End-effector Tracking Error Curves (no data)")
                    fig2.savefig(os.path.join(result_dir, "ee_6d_error_tracking.png"))
                    plt.close(fig2)

                print(f"📁 Trajectory plots saved to: {result_dir}")
            except Exception as e:
                print(f"❌ Failed to save trajectory plots: {e}")


def main():
    # 导入工具模块
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    args = utilities.parseConnectionArguments()

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        with utilities.DeviceConnection.createUdpConnection(args) as router_rt:
            base = BaseClient(router)
            base_cyclic = BaseCyclicClient(router_rt)

            planner = WaypointSpeedTrajectory(base, base_cyclic)
            # 1) 启动即到达初始姿态
            planner.move_to_initial_joint_angles()

            # 2) 等待单键 's' 开始（无需回车）
            print("\n📨 按 's' 键开始执行5秒间隔的轨迹，按 Ctrl+C 退出...")
            try:
                import termios, tty
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    while True:
                        ch = sys.stdin.read(1)
                        if ch.lower() == 's':
                            print("▶️ 接收到 's'，开始执行轨迹")
                            break
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                # 兼容环境不支持 raw 输入的情况，退化到需要回车
                cmd = input("输入 s 并回车开始: ").strip().lower()
                if cmd != 's':
                    print("未收到 's'，程序退出")
                    return

            # 3) 执行轨迹
            planner.run_waypoints()


if __name__ == "__main__":
    main()


