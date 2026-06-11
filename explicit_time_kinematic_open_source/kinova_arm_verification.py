#!/usr/bin/env python3
# filepath: /home/kinova/workspace/Kinova-kortex2_Gen3_G3L/api_python/examples/Explicit_time_kinematic/kinova_arm_verification.py

import time
import numpy as np
from Kinematic_fcn import Kinematic  # 正运动学函数
from planning_main import planning                 # 主函数
from constant import z_tool          # 工具长度常数
import pdb
import threading
import os
from datetime import datetime
import matplotlib
# 使用非交互式后端，确保无显示环境也能保存图片
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Kortex API 相关
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.SessionManager import SessionManager
from utilities import DeviceConnection, parseConnectionArguments

TIMEOUT_DURATION = 20

class KinovaJointPlanner:
    """Kinova机械臂关节角度规划器（集成版）"""
    
    def __init__(self, control_freq=1000):
        """
        初始化规划器
        
        Args:
            control_freq (int): 控制频率 (Hz)
        """
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self.integrated_theta = np.zeros(7)
        
    def reset_integrated_theta(self, initial_angles=None):
        """
        重置积分角度
        
        Args:
            initial_angles (np.array): 初始关节角度 (度)，如果为None则重置为零
        """
        if initial_angles is not None:
            self.integrated_theta = np.array(initial_angles).copy()
        else:
            self.integrated_theta = np.zeros(7)
    
    def plan_joint_angles(self, target_position, target_orientation, current_joint_angles, 
                         base_client=None, num_steps=1, reset_integration=True):
        """
        根据期望末端位置和姿态规划关节角度
        
        Args:
            target_position (np.array): 期望末端位置 [x, y, z] (米)
            target_orientation (np.array): 期望末端姿态 [rx, ry, rz] (弧度)
            current_joint_angles (np.array): 当前关节角度 (弧度)
            base_client: Kortex API的BaseClient实例 (可选)
            num_steps (int): 规划步数
            reset_integration (bool): 是否重置积分初始值为当前关节角度
            
        Returns:
            dict: 包含规划结果的字典
                - 'integrated_theta': 积分得到的期望关节角度 (度)
                - 'theta_dot_planning': 规划的关节角速度 (度/秒)
                - 'current_end_pose': 当前末端位姿 (如果提供base_client)
        """
        results = {
            'integrated_theta': None,
            'theta_dot_planning': None,
            'current_end_pose': None
        }
        
        # 将当前关节角度转换为度数，并设置为积分初始值
        if reset_integration:
            current_joint_angles_degree = np.rad2deg(current_joint_angles)
            self.reset_integrated_theta(current_joint_angles_degree)
            print(f"积分初始值设置为当前关节角度: {np.round(current_joint_angles_degree, 2)} 度")
        
        # 用于循环规划的关节角度变量
        planning_joint_angles = current_joint_angles.copy()
        
        for step in range(num_steps):
            # 调用主函数规划角速度
            theta_dot_planning = planning(target_position, target_orientation, planning_joint_angles)
            
            # 转换为度/秒并限制速度
            theta_dot_planning_degree = np.rad2deg(theta_dot_planning)
            for i in range(7):
                theta_dot_planning_degree[i] = np.clip(theta_dot_planning_degree[i], -10, 10)
            
            # 积分得到期望关节角度 (从当前关节角度开始积分)
            self.integrated_theta += np.array(theta_dot_planning_degree) * self.dt
            
            # 更新规划用的关节角度
            planning_joint_angles += theta_dot_planning * self.dt
            
            results['theta_dot_planning'] = theta_dot_planning_degree
            
            if step == 0:
                print(f"第{step+1}步规划: 角速度 = {np.round(theta_dot_planning_degree, 2)} 度/秒")
        
        results['integrated_theta'] = self.integrated_theta.copy()
        
        # 如果提供了base_client，获取当前末端位姿
        if base_client is not None:
            try:
                cartesian_pose = base_client.GetMeasuredCartesianPose()
                current_position = np.array([cartesian_pose.x, cartesian_pose.y, cartesian_pose.z])
                current_orientation = np.array([cartesian_pose.theta_x, 
                                              cartesian_pose.theta_y, 
                                              cartesian_pose.theta_z])
                results['current_end_pose'] = {
                    'position': current_position,
                    'orientation': current_orientation
                }
            except Exception as e:
                print(f"获取末端位姿失败: {e}")
        
        return results

    def plan_joint_angles_continuous(self, target_position, target_orientation, current_joint_angles, 
                                   base_client=None, num_steps=1):
        """
        连续规划关节角度，不重置积分初始值
        用于连续轨迹规划
        """
        return self.plan_joint_angles(
            target_position=target_position,
            target_orientation=target_orientation,
            current_joint_angles=current_joint_angles,
            base_client=base_client,
            num_steps=num_steps,
            reset_integration=False
        )


class PlotManager:
    """用于实时监控与保存曲线的绘图管理器"""

    def __init__(self, save_root_dir: str = "plots_output", update_every_loops: int = 20, save_snapshots: bool = False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(save_root_dir, f"run_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        self.update_every_loops = update_every_loops
        self.save_snapshots = save_snapshots

        # 数据缓冲区
        self.time_s = []
        self.joint_angle_error_deg = []  # shape: (N, 7)
        self.joint_speed_cmd_deg_s = []  # shape: (N, 7)
        self.ee_pos_err_m = []           # 标量: 位置误差范数
        self.ee_ori_err_rad = []         # 标量: 姿态误差范数

    @staticmethod
    def _normalize_deg(deg_values: np.ndarray) -> np.ndarray:
        # 归一化到[-180, 180)
        return ((deg_values + 180.0) % 360.0) - 180.0

    def log_step(self, t_s: float, current_theta_deg: np.ndarray, desired_theta_deg: np.ndarray,
                 theta_dot_cmd_deg_s: np.ndarray,
                 current_pos: np.ndarray, target_pos: np.ndarray,
                 current_ori: np.ndarray, target_ori: np.ndarray,
                 loop_count: int):
        # 关节角度误差（度），考虑环绕：对差值进行[-180,180)归一化
        angle_err = self._normalize_deg((desired_theta_deg - current_theta_deg).astype(float))

        # 末端位姿误差
        pos_err = float(np.linalg.norm(current_pos - target_pos))
        ori_err = float(np.linalg.norm(current_ori - target_ori))

        # 记录
        self.time_s.append(float(t_s))
        self.joint_angle_error_deg.append(angle_err.tolist())
        self.joint_speed_cmd_deg_s.append(theta_dot_cmd_deg_s.astype(float).tolist())
        self.ee_pos_err_m.append(pos_err)
        self.ee_ori_err_rad.append(ori_err)

        # 可选：周期性保存快照（默认关闭，仅保存最终曲线）
        if self.save_snapshots and self.update_every_loops > 0 and loop_count % self.update_every_loops == 0:
            self._save_all_figures(snapshot_suffix=f"_snapshot_{loop_count:06d}")

    def _save_all_figures(self, snapshot_suffix: str = ""):
        if len(self.time_s) == 0:
            return

        t = np.array(self.time_s)
        angle_err = np.array(self.joint_angle_error_deg)  # (N,7)
        speed_cmd = np.array(self.joint_speed_cmd_deg_s)  # (N,7)
        pos_err = np.array(self.ee_pos_err_m)
        ori_err = np.array(self.ee_ori_err_rad)

        # 图1: 关节角度误差曲线（每个关节一条）
        plt.figure(figsize=(11, 6))
        for j in range(7):
            plt.plot(t, angle_err[:, j], label=f"J{j+1}")
        plt.title("Joint Angle Error (deg)")
        plt.xlabel("Time (s)")
        plt.ylabel("Error (deg)")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=4)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"joint_angle_error{snapshot_suffix}.png"))
        plt.close()

        # 图2: 关节角速度指令曲线
        plt.figure(figsize=(11, 6))
        for j in range(7):
            plt.plot(t, speed_cmd[:, j], label=f"J{j+1}")
        plt.title("Joint Speed Command (deg/s)")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (deg/s)")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=4)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"joint_speed_cmd{snapshot_suffix}.png"))
        plt.close()

        # 图3: 末端位姿跟踪误差（位置/姿态范数）
        plt.figure(figsize=(11, 6))
        plt.plot(t, pos_err, label="Position error (m)")
        plt.plot(t, ori_err, label="Orientation error (rad)")
        plt.title("End-effector Tracking Error")
        plt.xlabel("Time (s)")
        plt.ylabel("Error")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"ee_tracking_error{snapshot_suffix}.png"))
        plt.close()

    def save_final(self):
        # 保存最终图与CSV数据
        self._save_all_figures(snapshot_suffix="")

        # CSV 保存
        try:
            t = np.array(self.time_s)[:, None]
            angle_err = np.array(self.joint_angle_error_deg)
            speed_cmd = np.array(self.joint_speed_cmd_deg_s)
            pos_err = np.array(self.ee_pos_err_m)[:, None]
            ori_err = np.array(self.ee_ori_err_rad)[:, None]

            joint_err_header = [f"angle_err_J{j+1}_deg" for j in range(7)]
            speed_cmd_header = [f"speed_cmd_J{j+1}_deg_s" for j in range(7)]
            header = ["time_s", *joint_err_header, *speed_cmd_header, "pos_err_m", "ori_err_rad"]
            data = np.hstack([t, angle_err, speed_cmd, pos_err, ori_err])
            csv_path = os.path.join(self.output_dir, "timeseries.csv")
            np.savetxt(csv_path, data, delimiter=",", header=",".join(header), comments="")
        except Exception as e:
            print(f"保存CSV失败: {e}")


def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications"""
    def check(notification, e = e):
        print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check

class KinovaArmController:
    """Kinova机械臂控制器"""
    
    def __init__(self, control_freq=2000):
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self.planner = KinovaJointPlanner(control_freq)
        self.current_target_index = 0
        self.targets = self._get_target_sequence()
        self.start_time = time.time()
        
    def _get_target_sequence(self):
        
        """先到指定初始位姿，再保持固定目标（到一个定值）"""
        return [
            {
                'name': 'INIT',
                # 初始位置/姿态（来自 YWtorque_test.py）
                'position': np.array([0.300, -0.089, 0.415]),
                'orientation': np.radians([142.1, -130.1, 20.7]),
                'duration': 5.0
            },
            {
                'name': 'SETPOINT',
                # 固定目标位置/姿态（来自原(297-298)）
                'position': np.array([0.396, -0.146, 0.159]),
                'orientation': np.radians([-54.4, -61.4, -127.2]),
                # 给予很大的持续时间，实质上恒定追踪
                'duration': 1e9
            }
        ]
    
    def get_current_target(self):
        """根据时间获取当前目标"""
        elapsed_time = time.time() - self.start_time
        cumulative_time = 0
        
        for i, target in enumerate(self.targets):
            cumulative_time += target['duration']
            if elapsed_time < cumulative_time:
                return i, target
        
        # 如果超过所有目标时间，返回最后一个目标
        return len(self.targets) - 1, self.targets[-1]
    
    def safety_check(self, joint_speeds):
        """安全检查关节速度"""
        max_speed = 10.0  # 度/秒
        for i, speed in enumerate(joint_speeds):
            if abs(speed) > max_speed:
                print(f"警告: 关节{i}速度{speed:.2f}超过安全限制{max_speed}")
                joint_speeds[i] = np.clip(speed, -max_speed, max_speed)
        return joint_speeds

def kortex_arm_verification():
    """Kinova机械臂验证主函数"""
    
    print("====== Kinova机械臂验证程序启动 ======")
    
    # 初始化控制器
    controller = KinovaArmController(control_freq=2000)
    
    # 解析连接参数
    args = parseConnectionArguments()
    
    with DeviceConnection.createTcpConnection(args) as router:
        # 建立连接
        session_manager = SessionManager(router)
        base_client = BaseClient(router)
        base_cycle_client = BaseCyclicClient(router)
        
        print("✓ Kortex API连接成功")
        
        # 初始化变量
        loop_count = 0
        target_switched = False
        
        # 初始化绘图管理器（仅保存整体过程的最终曲线，不保存中间快照）
        plot_mgr = PlotManager(save_root_dir="plots_output", update_every_loops=0, save_snapshots=False)

        try:
            print("开始闭环控制...")
            
            while True:
                start_time = time.time()
                
                # 1. 获取当前目标
                target_index, current_target = controller.get_current_target()
                
                # 检查是否切换了目标
                if target_index != controller.current_target_index:
                    controller.current_target_index = target_index
                    target_switched = True
                    print(f"\n>>> 切换到 {current_target['name']}")
                    print(f"    位置: {np.round(current_target['position'], 4)}")
                    print(f"    姿态: {np.round(np.rad2deg(current_target['orientation']), 2)} 度")
                
                # 2. 获取当前关节角度
                joint_angles = base_client.GetMeasuredJointAngles()
                current_theta_degree = np.array([joint.value for joint in joint_angles.joint_angles[:7]])
                current_theta_radian = np.deg2rad(current_theta_degree)
                
                # 3. 使用改进的规划器计算期望关节角度
                if target_switched:
                    # 目标切换时，重置积分初始值
                    planning_result = controller.planner.plan_joint_angles(
                        target_position=current_target['position'],
                        target_orientation=current_target['orientation'],
                        current_joint_angles=current_theta_radian,
                        base_client=base_client,
                        reset_integration=True
                    )
                    target_switched = False
                else:
                    # 连续规划模式
                    planning_result = controller.planner.plan_joint_angles_continuous(
                        target_position=current_target['position'],
                        target_orientation=current_target['orientation'],
                        current_joint_angles=current_theta_radian,
                        base_client=base_client
                    )
                
                # 4. 获取规划的关节角速度
                theta_dot_planning_degree = planning_result['theta_dot_planning']
                
                # 5. 安全检查
                theta_dot_planning_degree = controller.safety_check(theta_dot_planning_degree)
                
                # 6. 发送关节速度命令
                joint_speeds = Base_pb2.JointSpeeds()
                for i in range(7):
                    js = joint_speeds.joint_speeds.add()
                    js.joint_identifier = i
                    js.value = float(theta_dot_planning_degree[i])
                    js.duration = 0  # 立即生效
                
                base_client.SendJointSpeedsCommand(joint_speeds)
                
                # 7. 获取当前末端位姿（用于监控）
                cartesian_pose = base_client.GetMeasuredCartesianPose()
                current_position = np.array([cartesian_pose.x, cartesian_pose.y, cartesian_pose.z])
                current_orientation = np.array([cartesian_pose.theta_x, 
                                               cartesian_pose.theta_y, 
                                               cartesian_pose.theta_z])
                
                # 8. 计算位置误差
                position_error = np.linalg.norm(current_position - current_target['position'])
                orientation_error = np.linalg.norm(current_orientation - current_target['orientation'])
                
                # 9. 记录与绘图（轻量）
                t_s = time.time() - controller.start_time
                plot_mgr.log_step(
                    t_s=t_s,
                    current_theta_deg=current_theta_degree,
                    desired_theta_deg=planning_result['integrated_theta'],
                    theta_dot_cmd_deg_s=theta_dot_planning_degree,
                    current_pos=current_position,
                    target_pos=current_target['position'],
                    current_ori=current_orientation,
                    target_ori=current_target['orientation'],
                    loop_count=loop_count
                )

                # 10. 每10个循环打印一次状态
                if loop_count % 10 == 0:
                    print(f"\n[Loop {loop_count}] 目标: {current_target['name']}")
                    print(f"当前关节角度: {np.round(current_theta_degree, 2)} 度")
                    print(f"期望关节角度: {np.round(planning_result['integrated_theta'], 2)} 度")
                    print(f"当前末端位置: {np.round(current_position, 4)}")
                    print(f"目标末端位置: {np.round(current_target['position'], 4)}")
                    print(f"位置误差: {position_error:.4f}m, 姿态误差: {orientation_error:.4f}rad")
                    print(f"关节角速度: {np.round(theta_dot_planning_degree, 2)} 度/秒")
                
                # 11. 控制循环频率
                loop_count += 1
                sleep_time = controller.dt - (time.time() - start_time)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print(f"警告: 控制循环超时 {-sleep_time:.4f}s")
                    
        except KeyboardInterrupt:
            print("\n>>> 用户中断，正在安全停止机械臂...")
            
            # 发送零速度命令停止机械臂
            joint_speeds = Base_pb2.JointSpeeds()
            for i in range(7):
                js = joint_speeds.joint_speeds.add()
                js.joint_identifier = i
                js.value = 0.0
                js.duration = 0
            
            base_client.SendJointSpeedsCommand(joint_speeds)
            print("✓ 机械臂已安全停止")
            # 保存最终图与数据
            try:
                plot_mgr.save_final()
                print(f"✓ 曲线已保存至: {plot_mgr.output_dir}")
            except Exception as e:
                print(f"保存曲线失败: {e}")
            
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            
            # 紧急停止
            try:
                joint_speeds = Base_pb2.JointSpeeds()
                for i in range(7):
                    js = joint_speeds.joint_speeds.add()
                    js.joint_identifier = i
                    js.value = 0.0
                    js.duration = 0
                base_client.SendJointSpeedsCommand(joint_speeds)
                print("✓ 紧急停止成功")
            except:
                print("❌ 紧急停止失败")
            finally:
                # 出错也保存已有曲线
                try:
                    plot_mgr.save_final()
                    print(f"✓ 曲线已保存至: {plot_mgr.output_dir}")
                except Exception as e2:
                    print(f"保存曲线失败: {e2}")


if __name__ == "__main__":
    print("Kinova机械臂验证程序")
    
    choice = input("5个目标点开始 (s): ").strip()
    
    if choice == "s":
        kortex_arm_verification()
    else:
        print("无效选择")