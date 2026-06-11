import os
import sys
import time
import math
import numpy as np

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

# -----------------------------------------------------------------------------
# 全局参数集中定义（统一修改入口）
# -----------------------------------------------------------------------------
FREQUENCY_HZ = 2000.0                 # 控制频率 2000 Hz
DT = 1.0 / FREQUENCY_HZ               # 控制周期
Z_TOOL = -0.16746                     # 工具长度（米）
POS_TOLERANCE_M = 0.0001              # 位置精度: 0.1 mm
ANGLE_TOLERANCE_RAD = 0.1 * math.pi / 180.0  # 姿态精度: 0.1 度
MAX_JOINT_SPEED_DEG_S = 20.0          # 关节角速度上限 (deg/s)
MAX_JOINT_SPEED_RAD_S = math.radians(MAX_JOINT_SPEED_DEG_S)  # (rad/s)
SPEED_SCALE = 0.8                     # 速度缩放因子
MAX_TOTAL_SECONDS = 20               # 最长运行时间
PRINT_INTERVAL_CYCLES = int(0.5 / DT) # 每0.5秒打印一次状态

# 目标末端位姿（与 pose_end_planning.py 保持一致）
P_TARGET = np.array([0.389, -0.136, 0.214])
PHI_TARGET_DEG = np.array([-47.7, -65.2, -132.6])
PHI_TARGET = np.deg2rad(PHI_TARGET_DEG)


def Kinematic(theta, z_tool):
    # 占位导入（运行时将从同目录导入真实实现）
    from Kinematic_fcn import Kinematic as KinematicImpl
    return KinematicImpl(theta, z_tool)


def planning_wrapper(P_target, Phi_target, theta):
    # 占位导入（运行时将从同目录导入真实实现）
    from planning_main import planning
    return planning(P_target, Phi_target, theta)


class OnlineJointSpeedControl:
    def __init__(self, base: BaseClient, base_cyclic: BaseCyclicClient):
        self.base = base
        self.base_cyclic = base_cyclic

        self.device_manager = DeviceManagerClient(self.base.router)

        # Servo 模式
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        # 轴数
        self.actuator_count = self.base.GetActuatorCount().count
        if self.actuator_count != 7:
            raise ValueError(f"需要7轴机械臂，当前为 {self.actuator_count} 轴")

        # 状态
        self.iteration = 0
        self.start_time = time.time()

        print("✅ 在线关节速度控制初始化完成")
        print(f"🎯 目标位置: {P_TARGET}")
        print(f"🎯 目标姿态: {PHI_TARGET_DEG}°")
        print(f"⏱️  控制频率: {FREQUENCY_HZ:.0f} Hz (dt={DT*1000:.3f} ms)")

    def get_actual_joint_angles(self):
        fb = self.base_cyclic.RefreshFeedback()
        angles_deg = [fb.actuators[i].position for i in range(self.actuator_count)]
        return np.deg2rad(np.array(angles_deg))

    def compute_pose(self, theta):
        P, Phi, _, _ = Kinematic(theta, Z_TOOL)
        return P, Phi

    def compute_errors(self, P, Phi):
        pos_err = np.linalg.norm(P - P_TARGET)
        ang_err = np.linalg.norm(Phi - PHI_TARGET)
        return pos_err, ang_err

    def converged(self, pos_err, ang_err):
        return (pos_err < POS_TOLERANCE_M) and (ang_err < ANGLE_TOLERANCE_RAD)

    def compute_joint_speed_command(self, theta_current):
        # 由规划返回角速度估计（单位：rad/s）
        theta_dot = np.array(planning_wrapper(P_TARGET, PHI_TARGET, theta_current))
        theta_dot = np.clip(theta_dot, -MAX_JOINT_SPEED_RAD_S, MAX_JOINT_SPEED_RAD_S)
        theta_dot *= SPEED_SCALE
        return theta_dot

    def send_joint_speeds(self, theta_dot_rad_s):
        js = Base_pb2.JointSpeeds()
        for j in range(self.actuator_count):
            sp = js.joint_speeds.add()
            sp.joint_identifier = j
            sp.value = math.degrees(theta_dot_rad_s[j])  # API 需要度每秒
            sp.duration = 0
        self.base.SendJointSpeedsCommand(js)

    def print_status(self, pos_err, ang_err, theta_dot):
        if self.iteration % PRINT_INTERVAL_CYCLES != 0:
            return
        print(f"it={self.iteration:8d} | pos_err={pos_err*1000:8.4f} mm | ang_err={math.degrees(ang_err):8.4f} deg | |theta_dot|={np.linalg.norm(theta_dot):.3f} rad/s")

    def run(self):
        total_start = time.time()
        # 历史记录
        time_history = []
        joint_angle_history = []  # (rad)
        pos_error_history = []    # (m, scalar norm)
        angle_error_history = []  # (rad, scalar norm)
        pose_error_6d_history = []  # [dx, dy, dz, dphi_x, dphi_y, dphi_z]
        try:
            while True:
                loop_start = time.perf_counter()

                theta = self.get_actual_joint_angles()
                P, Phi = self.compute_pose(theta)
                pos_err, ang_err = self.compute_errors(P, Phi)
                if self.converged(pos_err, ang_err):
                    print("🎉 达到精度要求，停止速度控制")
                    break

                theta_dot = self.compute_joint_speed_command(theta)
                self.send_joint_speeds(theta_dot)

                self.print_status(pos_err, ang_err, theta_dot)
                self.iteration += 1

                # 记录历史
                time_history.append(time.time() - self.start_time)
                joint_angle_history.append(theta.copy())
                pos_error_history.append(pos_err)
                angle_error_history.append(ang_err)
                # 记录6维误差: 位置差(米) + 姿态差(弧度)
                pose_error_6d_history.append(np.concatenate([P - P_TARGET, Phi - PHI_TARGET]))

                # 节拍对齐 2000 Hz
                elapsed = time.perf_counter() - loop_start
                sleep_t = max(0.0, DT - elapsed)
                if sleep_t > 0:
                    time.sleep(sleep_t)

                if time.time() - total_start > MAX_TOTAL_SECONDS:
                    print("⏹️ 达到最大运行时长，停止速度控制")
                    break
        finally:
            try:
                self.base.Stop()
            except Exception:
                pass

            # 保存实时曲线
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                base_dir = os.path.join(os.path.dirname(__file__), "speed_results")
                os.makedirs(base_dir, exist_ok=True)
                result_dir = os.path.join(base_dir, f"run-{timestamp}")
                os.makedirs(result_dir, exist_ok=True)

                t = np.array(time_history)
                if len(joint_angle_history) > 0:
                    th = np.array(joint_angle_history)  # (N, 7) rad
                    # 图1：关节角度（度）
                    fig1, ax1 = plt.subplots(figsize=(12, 6))
                    for j in range(min(th.shape[1], 7)):
                        ax1.plot(t, np.rad2deg(th[:, j]), label=f"关节{j+1}")
                    ax1.set_title("实时关节角度曲线")
                    ax1.set_xlabel("时间 (s)")
                    ax1.set_ylabel("关节角度 (deg)")
                    ax1.grid(True, alpha=0.3)
                    ax1.legend(ncol=2)
                    fig1.tight_layout()
                    fig1_path = os.path.join(result_dir, "joint_angles.png")
                    fig1.savefig(fig1_path)
                    plt.close(fig1)

                # 图2：末端6维误差（dx, dy, dz, dphi_x, dphi_y, dphi_z），大图不画阈值线
                if len(pose_error_6d_history) > 0:
                    err6 = np.array(pose_error_6d_history)  # (N, 6)
                    pos_mm = err6[:, 0:3] * 1000.0  # 转mm
                    ang_deg = np.degrees(err6[:, 3:6])  # 转deg

                    fig2, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
                    fig2.suptitle('实时末端6维误差（大图不含阈值线）')
                    labels_pos = ['dx (mm)', 'dy (mm)', 'dz (mm)']
                    labels_ang = ['dphi_x (deg)', 'dphi_y (deg)', 'dphi_z (deg)']
                    # 位置误差
                    for i in range(3):
                        ax = axes[0, i]
                        ax.plot(t, pos_mm[:, i], 'b-')
                        ax.set_ylabel(labels_pos[i])
                        ax.grid(True, alpha=0.3)
                        # 中图：±0.2mm 放大
                        inset = inset_axes(ax, width="40%", height="40%", loc='upper right')
                        inset.plot(t, pos_mm[:, i], 'b-')
                        inset.set_ylim(-0.2, 0.2)
                        inset.grid(True, alpha=0.3)
                    # 姿态误差
                    for i in range(3):
                        ax = axes[1, i]
                        ax.plot(t, ang_deg[:, i], 'g-')
                        ax.set_ylabel(labels_ang[i])
                        ax.set_xlabel('时间 (s)')
                        ax.grid(True, alpha=0.3)
                        # 中图：±0.1度 放大
                        inset = inset_axes(ax, width="40%", height="40%", loc='upper right')
                        inset.plot(t, ang_deg[:, i], 'g-')
                        inset.set_ylim(-0.1, 0.1)
                        inset.grid(True, alpha=0.3)

                    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
                    fig2_path = os.path.join(result_dir, "ee_6d_error_tracking.png")
                    fig2.savefig(fig2_path)
                    plt.close(fig2)

                print(f"📁 实时曲线已保存至: {result_dir}")
            except Exception as e:
                print(f"❌ 保存实时曲线失败: {e}")


def main():
    # 导入工具模块
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    args = utilities.parseConnectionArguments()

    # 连接设备
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        with utilities.DeviceConnection.createUdpConnection(args) as router_rt:
            base = BaseClient(router)
            base_cyclic = BaseCyclicClient(router_rt)

            controller = OnlineJointSpeedControl(base, base_cyclic)
            input("\n🚀 按Enter键启动2000Hz关节角速度控制，Ctrl+C退出...")
            controller.run()


if __name__ == "__main__":
    main()


