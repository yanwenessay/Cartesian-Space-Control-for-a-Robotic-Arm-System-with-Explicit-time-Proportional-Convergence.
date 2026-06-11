import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import math
import threading
from planning_main import planning  
from Kinematic_fcn import Kinematic

# 导入Kortex API相关模块
from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.messages import ActuatorConfig_pb2, Base_pb2, BaseCyclic_pb2, Common_pb2
from kortex_api.RouterClient import RouterClientSendOptions

class RealTimeKinovaControl:
    """
    实时闭环Kinova机械臂积分控制系统
    持续从机械臂获取实际关节角度，通过planning计算角速度，积分后发送目标角度命令
    """
    
    def __init__(self, router, router_real_time):
        """
        初始化实时控制器
        """
        print("🚀 初始化实时闭环Kinova控制系统...")
        
        # Kortex API客户端初始化
        self.device_manager = DeviceManagerClient(router)
        self.actuator_config = ActuatorConfigClient(router)
        self.base = BaseClient(router)
        self.base_cyclic = BaseCyclicClient(router_real_time)
        
        # 通信消息结构初始化
        self.base_command = BaseCyclic_pb2.Command()
        self.base_feedback = BaseCyclic_pb2.Feedback()
        
        # 验证机械臂轴数
        self.actuator_count = self.base.GetActuatorCount().count
        if self.actuator_count != 7:
            raise ValueError(f"需要7轴机械臂，当前为 {self.actuator_count} 轴")
        
        # 设备配置
        self._setup_device_configuration()
        
        # 通信选项配置
        self.sendOption = RouterClientSendOptions()
        self.sendOption.andForget = False
        self.sendOption.delay_ms = 0
        self.sendOption.timeout_ms = 1
        
        # 控制参数
        self.z_tool = -0.16746  # 工具长度
        
        # 精度要求
        self.pos_tolerance = 0.0001    # 位置精度: 0.1mm
        self.angle_tolerance = 0.1 * math.pi / 180.0  # 姿态精度: 0.1度
        
        # 控制参数
        self.dt = 0.02                   # 控制周期 50Hz
        self.max_joint_velocity = 0.5   # 最大关节角速度限制
        self.velocity_scale_factor = 0.8 # 速度缩放因子
        
        # 目标位姿
        self.P_target = np.array([0.389, -0.136, 0.214])
        self.Phi_target_deg = np.array([-47.7, -65.2, -132.6])
        self.Phi_target = np.deg2rad(self.Phi_target_deg)
        
        # 控制状态
        self.control_active = False
        self.force_stop = False
        
        # 性能统计
        self.iteration_count = 0
        self.start_time = None
        self.best_pos_error = float('inf')
        self.best_angle_error = float('inf')
        
        print(f"✅ 实时控制器初始化完成")
        print(f"🎯 位置精度目标: {self.pos_tolerance*1000:.1f}mm")
        print(f"🎯 姿态精度目标: {math.degrees(self.angle_tolerance):.1f}°")
        print(f"⏱️  控制频率: {1/self.dt:.0f}Hz")
    
    def _setup_device_configuration(self):
        """设置设备配置"""
        device_handles = self.device_manager.ReadAllDevices()
        for handle in device_handles.device_handle:
            if handle.device_type == Common_pb2.BIG_ACTUATOR or handle.device_type == Common_pb2.SMALL_ACTUATOR:
                self.base_command.actuators.add()
                self.base_feedback.actuators.add()
    
    def get_actual_joint_angles(self):
        """
        实时获取机械臂当前实际关节角度
        
        Returns:
            numpy.array: 当前实际关节角度(弧度)
        """
        try:
            # 刷新反馈获取最新状态
            self.base_feedback = self.base_cyclic.RefreshFeedback()
            
            # 提取实际关节角度
            actual_angles_deg = []
            for i in range(self.actuator_count):
                angle_deg = self.base_feedback.actuators[i].position
                actual_angles_deg.append(angle_deg)
            
            # 转换为弧度
            actual_angles_rad = np.array(actual_angles_deg) * math.pi / 180.0
            
            return actual_angles_rad
            
        except Exception as e:
            print(f"❌ 获取实际关节角度失败: {e}")
            return None
    
    def calculate_current_pose(self, theta_actual):
        """
        计算当前末端位姿
        
        Args:
            theta_actual: 实际关节角度
            
        Returns:
            tuple: (位置, 姿态, 成功标志)
        """
        try:
            P_current, Phi_current, _, _ = Kinematic(theta_actual, self.z_tool)
            return P_current, Phi_current, True
        except Exception as e:
            print(f"❌ 正运动学计算失败: {e}")
            return np.zeros(3), np.zeros(3), False
    
    def calculate_pose_error(self, P_current, Phi_current):
        """
        计算位姿误差
        
        Returns:
            tuple: (位置误差, 姿态误差)
        """
        pos_error = np.linalg.norm(P_current - self.P_target)
        angle_error = np.linalg.norm(Phi_current - self.Phi_target)
        return pos_error, angle_error
    
    def is_converged(self, pos_error, angle_error):
        """检查是否收敛到目标精度"""
        pos_converged = pos_error < self.pos_tolerance
        angle_converged = angle_error < self.angle_tolerance
        
        # 更新最佳误差记录
        if pos_error < self.best_pos_error:
            self.best_pos_error = pos_error
        if angle_error < self.best_angle_error:
            self.best_angle_error = angle_error
        
        return pos_converged and angle_converged
    
    def compute_target_joint_angles(self, theta_actual):
        """
        基于实际关节角度计算目标关节角度
        
        Args:
            theta_actual: 当前实际关节角度(弧度)
            
        Returns:
            numpy.array: 目标关节角度(弧度)
        """
        try:
            # 调用planning函数，输入实际关节角度
            theta_dot_planning = planning(self.P_target, self.Phi_target, theta_actual)
            
            # 转换为numpy数组并限制速度
            theta_dot_planning = np.array(theta_dot_planning)
            theta_dot_limited = np.clip(theta_dot_planning, 
                                      -self.max_joint_velocity, 
                                      self.max_joint_velocity)
            
            # 应用速度缩放因子
            theta_dot_scaled = theta_dot_limited * self.velocity_scale_factor
            
            # 积分计算目标关节角度
            theta_target = theta_actual + theta_dot_scaled * self.dt
            
            return theta_target
            
        except Exception as e:
            print(f"❌ 目标角度计算失败: {e}")
            return theta_actual  # 返回当前角度作为保护
    
    def send_joint_angle_command(self, target_angles_rad):
        """
        发送关节角度命令到机械臂
        
        Args:
            target_angles_rad: 目标关节角度(弧度)
            
        Returns:
            bool: 是否成功发送命令
        """
        try:
            # 确保处于单层伺服模式（一次设置足够，偶尔重设以稳妥）
            base_servo_mode = Base_pb2.ServoingModeInformation()
            base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
            self.base.SetServoingMode(base_servo_mode)

            # 模仿 01-move_angular_and_cartesian.py 构造高层关节角度动作并发送
            target_angles_deg = [math.degrees(angle) for angle in target_angles_rad]

            action = Base_pb2.Action()
            action.name = "Realtime reach joint angles"
            action.application_data = ""

            for joint_id in range(self.actuator_count):
                joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
                joint_angle.joint_identifier = joint_id
                joint_angle.value = target_angles_deg[joint_id]

            # 异步执行动作（不等待结束，以保持循环实时性）
            self.base.ExecuteAction(action)

            return True

        except Exception as e:
            print(f"❌ 发送关节命令失败: {e}")
            return False
    
    def print_status(self, iteration, pos_error, angle_error, P_current, Phi_current, 
                    theta_actual, elapsed_time):
        """打印实时控制状态"""
        pos_error_mm = pos_error * 1000
        angle_error_deg = math.degrees(angle_error)
        
        # 计算收敛进度
        pos_progress = min(100, (self.pos_tolerance / pos_error) * 100) if pos_error > 0 else 100
        angle_progress = min(100, (self.angle_tolerance / angle_error) * 100) if angle_error > 0 else 100
        overall_progress = min(pos_progress, angle_progress)
        
        print(f"\n{'='*80}")
        print(f"🔄 实时控制 - 迭代 {iteration:6d} | ⏱️  {elapsed_time:8.1f}s | 🎯 进度: {overall_progress:5.1f}%")
        print(f"{'='*80}")
        
        print(f"📍 位置状态:")
        print(f"   当前误差: {pos_error_mm:8.4f}mm | 目标: <{self.pos_tolerance*1000:.1f}mm | 进度: {pos_progress:5.1f}%")
        print(f"   历史最佳: {self.best_pos_error*1000:8.4f}mm")
        print(f"   当前位置: [{P_current[0]:7.4f}, {P_current[1]:7.4f}, {P_current[2]:7.4f}]")
        print(f"   目标位置: [{self.P_target[0]:7.4f}, {self.P_target[1]:7.4f}, {self.P_target[2]:7.4f}]")
        
        print(f"\n📐 姿态状态:")
        print(f"   当前误差: {angle_error_deg:8.4f}° | 目标: <{math.degrees(self.angle_tolerance):.1f}° | 进度: {angle_progress:5.1f}%")
        print(f"   历史最佳: {math.degrees(self.best_angle_error):8.4f}°")
        print(f"   当前姿态: [{math.degrees(Phi_current[0]):6.2f}°, "
              f"{math.degrees(Phi_current[1]):6.2f}°, {math.degrees(Phi_current[2]):6.2f}°]")
        print(f"   目标姿态: [{self.Phi_target_deg[0]:6.2f}°, "
              f"{self.Phi_target_deg[1]:6.2f}°, {self.Phi_target_deg[2]:6.2f}°]")
        
        print(f"\n🤖 关节角度 (实际):")
        actual_angles_deg = [math.degrees(angle) for angle in theta_actual]
        print(f"   关节1-4: [{actual_angles_deg[0]:6.2f}°, {actual_angles_deg[1]:6.2f}°, "
              f"{actual_angles_deg[2]:6.2f}°, {actual_angles_deg[3]:6.2f}°]")
        print(f"   关节5-7: [{actual_angles_deg[4]:6.2f}°, {actual_angles_deg[5]:6.2f}°, "
              f"{actual_angles_deg[6]:6.2f}°]")
        
        print(f"\n⚙️  控制信息:")
        print(f"   控制频率: {1/self.dt:.0f}Hz | 时间步长: {self.dt:.3f}s")
        print(f"   速度缩放: {self.velocity_scale_factor:.3f} | 平均周期: {elapsed_time/max(1,iteration):.4f}s")
        
        # 收敛状态提示
        if pos_error < self.pos_tolerance and angle_error < self.angle_tolerance:
            print(f"\n🎉 已达到目标精度！")
        elif pos_error < self.pos_tolerance * 2 and angle_error < self.angle_tolerance * 2:
            print(f"\n🔥 非常接近目标，继续精细调整...")
        elif pos_error < self.pos_tolerance * 10 and angle_error < self.angle_tolerance * 10:
            print(f"\n⚡ 快速收敛中...")
        else:
            print(f"\n🚀 向目标移动中...")
    
    def run_realtime_control(self, print_interval=25):
        """
        运行实时闭环控制 - 持续到精度达到
        
        核心流程:
        1. 获取实际关节角度 theta_actual
        2. 调用 planning(P_target, Phi_target, theta_actual) 得到 theta_dot
        3. 积分: theta_target = theta_actual + theta_dot * dt  
        4. 发送 theta_target 命令到机械臂
        5. 重复直到精度达到
        """
        
        print(f"\n" + "🚀"*30)
        print("🎯 启动实时闭环积分控制")
        print("🚀"*30)
        print(f"📍 目标位置: {self.P_target}")
        print(f"📐 目标姿态: {self.Phi_target_deg}°")
        print(f"🎯 位置精度: <{self.pos_tolerance*1000:.1f}mm")
        print(f"🎯 姿态精度: <{math.degrees(self.angle_tolerance):.1f}°")
        print(f"⏱️  控制频率: {1/self.dt:.0f}Hz")
        print(f"🔄 实时闭环: 获取实际角度 → planning → 积分 → 发送命令")
        print("🚀"*30)
        
        # 初始化控制
        self.control_active = True
        self.force_stop = False
        self.start_time = time.time()
        self.iteration_count = 0
        
        # 历史记录
        time_history = []
        pos_error_history = []
        angle_error_history = []
        theta_actual_history = []
        theta_target_history = []
        
        converged = False
        
        try:
            while self.control_active and not self.force_stop and not converged:
                
                cycle_start_time = time.time()
                
                # 步骤1: 获取机械臂实际关节角度
                theta_actual = self.get_actual_joint_angles()
                if theta_actual is None:
                    print(f"⚠️ 无法获取实际关节角度，跳过本次循环")
                    time.sleep(self.dt)
                    continue
                
                # 步骤2: 计算当前实际末端位姿
                P_current, Phi_current, pose_success = self.calculate_current_pose(theta_actual)
                if not pose_success:
                    print(f"⚠️ 位姿计算失败，跳过本次循环")
                    time.sleep(self.dt)
                    continue
                
                # 步骤3: 计算位姿误差
                pos_error, angle_error = self.calculate_pose_error(P_current, Phi_current)
                
                # 步骤4: 检查收敛
                if self.is_converged(pos_error, angle_error):
                    converged = True
                    print(f"\n🎉🎉🎉 实时控制成功收敛！🎉🎉🎉")
                    print(f"✅ 最终位置误差: {pos_error*1000:.6f}mm (< {self.pos_tolerance*1000:.1f}mm)")
                    print(f"✅ 最终姿态误差: {math.degrees(angle_error):.6f}° (< {math.degrees(self.angle_tolerance):.1f}°)")
                    print(f"✅ 总迭代次数: {self.iteration_count}")
                    print(f"✅ 总耗时: {time.time() - self.start_time:.2f}s")
                    break
                
                # 步骤5: 调用planning计算目标关节角度
                theta_target = self.compute_target_joint_angles(theta_actual)
                
                # 步骤6: 发送关节角度命令
                command_success = self.send_joint_angle_command(theta_target)
                if not command_success:
                    print(f"⚠️ 命令发送失败，重试...")
                    time.sleep(self.dt * 0.5)
                    continue
                
                # 记录历史数据
                elapsed_time = time.time() - self.start_time
                time_history.append(elapsed_time)
                pos_error_history.append(pos_error)
                angle_error_history.append(angle_error)
                theta_actual_history.append(theta_actual.copy())
                theta_target_history.append(theta_target.copy())
                
                # 定期打印状态
                if self.iteration_count % print_interval == 0:
                    self.print_status(self.iteration_count, pos_error, angle_error, 
                                    P_current, Phi_current, theta_actual, elapsed_time)
                
                self.iteration_count += 1
                
                # 控制周期时间管理
                cycle_time = time.time() - cycle_start_time
                sleep_time = max(0, self.dt - cycle_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif cycle_time > self.dt * 1.5:
                    print(f"⚠️ 控制周期超时: {cycle_time:.4f}s > {self.dt:.4f}s")
        
        except KeyboardInterrupt:
            print(f"\n⚠️ 用户强制停止实时控制")
            self.force_stop = True
        except Exception as e:
            print(f"\n❌ 实时控制异常: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.control_active = False
            
            # 最终验证 - 再次获取实际状态
            print(f"\n📊 最终验证...")
            final_theta_actual = self.get_actual_joint_angles()
            if final_theta_actual is not None:
                P_final, Phi_final, _ = self.calculate_current_pose(final_theta_actual)
                final_pos_error, final_angle_error = self.calculate_pose_error(P_final, Phi_final)
                
                # 结果汇总
                result = {
                    'converged': converged,
                    'force_stopped': self.force_stop,
                    'final_theta_actual': final_theta_actual,
                    'final_position': P_final,
                    'final_orientation': Phi_final,
                    'final_pos_error': final_pos_error,
                    'final_angle_error': final_angle_error,
                    'total_iterations': self.iteration_count,
                    'total_time': time.time() - self.start_time,
                    'best_pos_error': self.best_pos_error,
                    'best_angle_error': self.best_angle_error,
                    'time_history': np.array(time_history),
                    'pos_error_history': pos_error_history,
                    'angle_error_history': angle_error_history,
                    'theta_actual_history': np.array(theta_actual_history),
                    'theta_target_history': np.array(theta_target_history),
                    'target_position': self.P_target,
                    'target_orientation': self.Phi_target
                }
                
                # 打印最终结果
                self.print_final_results(result)
                
                # 绘制结果
                if len(time_history) > 1:
                    self.plot_realtime_results(result)
                
                return result
            else:
                print(f"❌ 无法获取最终状态")
                return None
    
    def print_final_results(self, result):
        """打印最终控制结果"""
        print(f"\n" + "🎯"*40)
        print("📊 实时控制最终结果")
        print("🎯"*40)
        
        if result['converged']:
            status = "🎉 完美收敛 - 达到目标精度"
        elif result['force_stopped']:
            status = "⚠️ 用户强制停止"
        else:
            status = "❌ 异常终止"
        
        print(f"状态: {status}")
        print(f"总迭代次数: {result['total_iterations']:,}")
        print(f"总耗时: {result['total_time']:.2f}s ({result['total_time']/60:.1f}分钟)")
        print(f"实际控制频率: {result['total_iterations']/result['total_time']:.1f}Hz")
        print(f"平均控制周期: {result['total_time']/max(1, result['total_iterations']):.4f}s")
        
        print(f"\n📍 位置结果:")
        print(f"  目标位置: {result['target_position']}")
        print(f"  最终位置: {result['final_position']}")
        print(f"  最终误差: {result['final_pos_error']*1000:.6f}mm")
        print(f"  历史最佳: {result['best_pos_error']*1000:.6f}mm")
        print(f"  精度要求: <{self.pos_tolerance*1000:.1f}mm")
        pos_achieved = "✅ 已达到" if result['final_pos_error'] < self.pos_tolerance else "❌ 未达到"
        print(f"  达成状态: {pos_achieved}")
        
        print(f"\n📐 姿态结果:")
        print(f"  目标姿态: {np.rad2deg(result['target_orientation'])}°")
        print(f"  最终姿态: {np.rad2deg(result['final_orientation'])}°")
        print(f"  最终误差: {math.degrees(result['final_angle_error']):.6f}°")
        print(f"  历史最佳: {math.degrees(result['best_angle_error']):.6f}°")
        print(f"  精度要求: <{math.degrees(self.angle_tolerance):.1f}°")
        angle_achieved = "✅ 已达到" if result['final_angle_error'] < self.angle_tolerance else "❌ 未达到"
        print(f"  达成状态: {angle_achieved}")
        
        overall_success = (result['converged'] and 
                          result['final_pos_error'] < self.pos_tolerance and 
                          result['final_angle_error'] < self.angle_tolerance)
        overall_status = "🎉 完全成功" if overall_success else "⚠️ 需要继续"
        print(f"\n🏆 总体状态: {overall_status}")
        print("🎯"*40)
    
    def plot_realtime_results(self, result):
        """绘制实时控制结果"""
        try:
            plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('实时闭环Kinova机械臂控制结果', fontsize=16, fontweight='bold')
            
            time_hist = result['time_history']
            pos_errors_mm = np.array(result['pos_error_history']) * 1000
            angle_errors_deg = np.rad2deg(result['angle_error_history'])
            iterations = range(len(pos_errors_mm))
            
            # 1. 位置误差收敛
            ax1 = axes[0, 0]
            ax1.semilogy(time_hist, pos_errors_mm, 'b-', linewidth=2, label='位置误差')
            ax1.axhline(y=self.pos_tolerance*1000, color='r', linestyle='--', 
                       linewidth=3, label=f'目标精度 ({self.pos_tolerance*1000:.1f}mm)')
            ax1.set_xlabel('时间 (s)')
            ax1.set_ylabel('位置误差 (mm, 对数尺度)')
            ax1.set_title('位置误差实时收敛')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 姿态误差收敛
            ax2 = axes[0, 1]
            ax2.semilogy(time_hist, angle_errors_deg, 'g-', linewidth=2, label='姿态误差')
            ax2.axhline(y=math.degrees(self.angle_tolerance), color='r', linestyle='--',
                       linewidth=3, label=f'目标精度 ({math.degrees(self.angle_tolerance):.1f}°)')
            ax2.set_xlabel('时间 (s)')
            ax2.set_ylabel('姿态误差 (°, 对数尺度)')
            ax2.set_title('姿态误差实时收敛')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. 实际vs目标关节角度对比 (最后1000个点)
            ax3 = axes[0, 2]
            if len(result['theta_actual_history']) > 0:
                # 显示最后1000个数据点
                n_show = min(1000, len(result['theta_actual_history']))
                show_actual = result['theta_actual_history'][-n_show:]
                show_target = result['theta_target_history'][-n_show:]
                show_time = time_hist[-n_show:]
                
                # 只显示前3个关节角度
                for i in range(min(3, show_actual.shape[1])):
                    ax3.plot(show_time, np.rad2deg(show_actual[:, i]), 
                            'b-', alpha=0.7, label=f'实际关节{i+1}' if i == 0 else "")
                    ax3.plot(show_time, np.rad2deg(show_target[:, i]), 
                            'r--', alpha=0.7, label=f'目标关节{i+1}' if i == 0 else "")
                
                ax3.set_xlabel('时间 (s)')
                ax3.set_ylabel('关节角度 (°)')
                ax3.set_title('实际vs目标关节角度 (最后1000点)')
                ax3.legend(['实际', '目标'])
                ax3.grid(True, alpha=0.3)
            
            # 4. 控制频率分析
            ax4 = axes[1, 0]
            if len(time_hist) > 1:
                control_periods = np.diff(time_hist)
                control_freq = 1.0 / control_periods
                ax4.plot(time_hist[1:], control_freq, 'purple', linewidth=1, alpha=0.7)
                ax4.axhline(y=1/self.dt, color='r', linestyle='--', linewidth=2, 
                           label=f'目标频率 ({1/self.dt:.0f}Hz)')
                ax4.set_xlabel('时间 (s)')
                ax4.set_ylabel('实际控制频率 (Hz)')
                ax4.set_title('实时控制频率')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            # 5. 误差线性对比
            ax5 = axes[1, 1]
            ax5.plot(time_hist, pos_errors_mm, 'b-', linewidth=2, label='位置误差 (mm)')
            ax5_twin = ax5.twinx()
            ax5_twin.plot(time_hist, angle_errors_deg, 'g-', linewidth=2, label='姿态误差 (°)')
            ax5.axhline(y=self.pos_tolerance*1000, color='r', linestyle='--', alpha=0.7)
            ax5_twin.axhline(y=math.degrees(self.angle_tolerance), color='r', linestyle='--', alpha=0.7)
            ax5.set_xlabel('时间 (s)')
            ax5.set_ylabel('位置误差 (mm)', color='b')
            ax5_twin.set_ylabel('姿态误差 (°)', color='g')
            ax5.set_title('误差时间历程')
            ax5.grid(True, alpha=0.3)
            
            # 6. 控制性能统计
            ax6 = axes[1, 2]
            metrics = ['迭代数', '耗时(s)', '位置误差(mm)', '姿态误差(°)', '控制频率(Hz)']
            values = [
                result['total_iterations'],
                result['total_time'],
                result['final_pos_error'] * 1000,
                math.degrees(result['final_angle_error']),
                result['total_iterations'] / result['total_time']
            ]
            colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
            
            # 标准化数值用于显示
            normalized_values = []
            for i, val in enumerate(values):
                if i == 0:  # 迭代数
                    normalized_values.append(val / 1000)  # 以千为单位
                elif i == 1:  # 时间
                    normalized_values.append(val / 60)    # 以分钟为单位
                else:
                    normalized_values.append(val)
            
            bars = ax6.bar(range(len(metrics)), normalized_values, color=colors, alpha=0.8)
            ax6.set_xticks(range(len(metrics)))
            ax6.set_xticklabels(metrics, rotation=45, ha='right')
            ax6.set_title('实时控制性能指标')
            
            # 在柱状图上添加原始数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                if value >= 1000:
                    label = f'{value:.0f}'
                elif value >= 1:
                    label = f'{value:.1f}'
                else:
                    label = f'{value:.3f}'
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        label, ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ 绘图失败: {e}")
    
    def emergency_stop(self):
        """紧急停止"""
        print(f"\n🛑 紧急停止激活")
        self.force_stop = True
        self.control_active = False
        try:
            self.base.Stop()
        except:
            pass

    def plan_and_send_final_angle(self, max_iterations=5000, print_interval=50):
        """
        先在本地通过运动学与规划内环迭代，直接计算出满足精度的期望关节角度，
        然后一次性将最终角度发送给机械臂。
        """
        print(f"\n" + "🧠"*20)
        print("🎯 启动离线迭代规划：直接收敛到期望关节角后一次性下发")
        print("🧠"*20)

        # 获取当前实际关节角度作为起点
        theta_actual = self.get_actual_joint_angles()
        if theta_actual is None:
            print("❌ 无法获取实际关节角度，退出")
            return None

        theta_iter = theta_actual.copy()
        self.iteration_count = 0
        self.best_pos_error = float('inf')
        self.best_angle_error = float('inf')
        self.start_time = time.time()

        converged = False

        while self.iteration_count < max_iterations:
            # 正运动学获取当前位姿
            P_current, Phi_current, pose_ok = self.calculate_current_pose(theta_iter)
            if not pose_ok:
                print("⚠️ 位姿计算失败，终止规划")
                break

            # 误差评估
            pos_error, angle_error = self.calculate_pose_error(P_current, Phi_current)
            if self.is_converged(pos_error, angle_error):
                converged = True
                break

            # 规划角速度
            try:
                theta_dot = planning(self.P_target, self.Phi_target, theta_iter)
            except Exception as e:
                print(f"❌ planning 调用失败: {e}")
                break

            # 限幅 + 缩放 + 积分更新（本地迭代，不下发）
            theta_dot = np.clip(np.array(theta_dot), -self.max_joint_velocity, self.max_joint_velocity)
            theta_iter = theta_iter + theta_dot * self.dt * self.velocity_scale_factor

            # 打印进度
            if self.iteration_count % print_interval == 0:
                elapsed_time = time.time() - self.start_time
                self.print_status(self.iteration_count, pos_error, angle_error, P_current, Phi_current, theta_iter, elapsed_time)

            self.iteration_count += 1

        # 结果处理
        if not converged:
            print(f"⚠️ 未在 {max_iterations} 次内达到精度，仍下发当前最优解")

        # 下发最终角度（一次性动作）
        sent = self.send_joint_angle_command(theta_iter)
        if sent:
            print("✅ 最终角度已发送")
        else:
            print("❌ 最终角度发送失败")

        return {
            'converged': converged,
            'theta_final': theta_iter,
            'iterations': self.iteration_count,
            'elapsed': time.time() - self.start_time
        }

    def run_online_iterative_control(self, interval_s=0.5, inner_time_ratio=0.7, max_total_seconds=600, print_interval=5, result_dir=None):
        """
        在线模式：若误差不在容差内，则每 interval_s 秒进行一次局部迭代规划，
        计算新的期望角度并一次性发送。

        inner_time_ratio 决定每个周期用于本地迭代的时间比例，其余时间sleep保证节拍。
        """
        print(f"\n" + "🚀"*30)
        print("🎯 启动在线迭代规划发送：每 {0:.3f}s 计算并下发一次".format(interval_s))
        print("🚀"*30)

        self.control_active = True
        self.force_stop = False
        self.start_time = time.time()
        self.iteration_count = 0

        total_start = time.time()
        converged = False

        # 历史记录
        time_history = []
        pos_error_history = []
        angle_error_history = []
        joint_velocity_history = []  # 以 (theta_iter - theta_actual)/interval_s 近似

        try:
            while self.control_active and not self.force_stop:
                cycle_start = time.time()

                # 1) 读取当前真实关节角
                theta_actual = self.get_actual_joint_angles()
                if theta_actual is None:
                    print("⚠️ 无法获取实际角度，跳过本周期")
                    time.sleep(interval_s)
                    continue

                # 2) 评估当前误差
                P_current, Phi_current, pose_ok = self.calculate_current_pose(theta_actual)
                if not pose_ok:
                    print("⚠️ 位姿计算失败，跳过本周期")
                    time.sleep(interval_s)
                    continue

                pos_error, angle_error = self.calculate_pose_error(P_current, Phi_current)
                if self.is_converged(pos_error, angle_error):
                    converged = True
                    print("🎉 误差在容差内，停止在线控制")
                    break

                # 3) 在本地进行短时间的内环迭代以逼近目标角
                theta_iter = theta_actual.copy()
                inner_budget = interval_s * inner_time_ratio
                inner_start = time.time()
                inner_steps = 0
                while time.time() - inner_start < inner_budget:
                    try:
                        theta_dot = planning(self.P_target, self.Phi_target, theta_iter)
                    except Exception as e:
                        print(f"❌ planning 调用失败: {e}")
                        break
                    theta_dot = np.clip(np.array(theta_dot), -self.max_joint_velocity, self.max_joint_velocity)
                    theta_iter = theta_iter + theta_dot * self.dt * self.velocity_scale_factor
                    inner_steps += 1

                # 4) 下发本周期最终角度
                sent = self.send_joint_angle_command(theta_iter)
                if not sent:
                    print("⚠️ 本周期发送失败")

                # 5) 记录历史并打印节拍状态
                elapsed_time = time.time() - self.start_time
                time_history.append(elapsed_time)
                pos_error_history.append(pos_error)
                angle_error_history.append(angle_error)
                approx_theta_dot = (theta_iter - theta_actual) / max(1e-6, interval_s)
                joint_velocity_history.append(approx_theta_dot.copy())
                if self.iteration_count % print_interval == 0:
                    self.print_status(self.iteration_count, pos_error, angle_error, P_current, Phi_current, theta_iter, elapsed_time)

                self.iteration_count += 1

                # 6) 周期对齐
                cycle_elapsed = time.time() - cycle_start
                sleep_time = max(0.0, interval_s - cycle_elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # 7) 总时长保护
                if time.time() - total_start > max_total_seconds:
                    print("⏹️ 达到最大运行时间，停止在线控制")
                    break

        except KeyboardInterrupt:
            print("\n⚠️ 用户中断在线控制")
            self.force_stop = True
        except Exception as e:
            print(f"\n❌ 在线控制异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.control_active = False

            # 保存结果图
            try:
                import matplotlib.pyplot as plt
                # 结果目录
                if result_dir is None:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    base_dir = os.path.join(os.path.dirname(__file__), "online_results")
                    os.makedirs(base_dir, exist_ok=True)
                    result_dir = os.path.join(base_dir, f"run-{timestamp}")
                os.makedirs(result_dir, exist_ok=True)

                # 转换数组
                thist = np.array(time_history)
                perr_mm = np.array(pos_error_history) * 1000.0
                aerr_deg = np.rad2deg(np.array(angle_error_history))
                vel_hist = np.array(joint_velocity_history)  # shape: (N, 7)

                # 图1：末端位姿误差（位置mm、姿态deg）
                fig1, ax1 = plt.subplots(2, 1, figsize=(10, 8))
                fig1.suptitle('在线控制 - 末端位姿误差跟踪')
                if thist.size > 0:
                    ax1[0].plot(thist, perr_mm, 'b-', label='位置误差 (mm)')
                    ax1[0].axhline(self.pos_tolerance*1000.0, color='r', linestyle='--', label='目标 (mm)')
                    ax1[0].set_ylabel('位置误差 (mm)')
                    ax1[0].legend(); ax1[0].grid(True, alpha=0.3)
                    ax1[1].plot(thist, aerr_deg, 'g-', label='姿态误差 (deg)')
                    ax1[1].axhline(np.degrees(self.angle_tolerance), color='r', linestyle='--', label='目标 (deg)')
                    ax1[1].set_xlabel('时间 (s)'); ax1[1].set_ylabel('姿态误差 (deg)')
                    ax1[1].legend(); ax1[1].grid(True, alpha=0.3)
                fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig1_path = os.path.join(result_dir, 'pose_error_tracking.png')
                fig1.savefig(fig1_path)
                plt.close(fig1)

                # 图2：关节角速度曲线（7条）
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                fig2.suptitle('在线控制 - 关节角速度曲线')
                if vel_hist.size > 0 and thist.size == vel_hist.shape[0]:
                    for j in range(min(vel_hist.shape[1], self.actuator_count)):
                        ax2.plot(thist, vel_hist[:, j], label=f'关节{j+1}')
                ax2.set_xlabel('时间 (s)')
                ax2.set_ylabel('角速度 (rad/s)')
                ax2.grid(True, alpha=0.3)
                ax2.legend(ncol=2)
                fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig2_path = os.path.join(result_dir, 'joint_velocity_curves.png')
                fig2.savefig(fig2_path)
                plt.close(fig2)

                print(f"📁 结果已保存至: {result_dir}")
                print(f" - 位姿误差: {fig1_path}")
                print(f" - 角速度曲线: {fig2_path}")
            except Exception as e:
                print(f"❌ 保存图像失败: {e}")

def main():
    """
    主函数 - 实时闭环Kinova机械臂控制
    """
    import argparse
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities
    
    parser = argparse.ArgumentParser(description='Kinova机械臂实时闭环积分控制')
    args = utilities.parseConnectionArguments(parser)
    
    print("🤖 Kinova 7自由度机械臂实时闭环控制系统")
    print("="*80)
    
    # 建立连接
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        with utilities.DeviceConnection.createUdpConnection(args) as router_real_time:
            
            try:
                # 创建实时控制器
                controller = RealTimeKinovaControl(router, router_real_time)
                
                # 显示当前状态
                print(f"\n📊 当前机械臂状态:")
                current_angles = controller.get_actual_joint_angles()
                if current_angles is not None:
                    P_current, Phi_current, _ = controller.calculate_current_pose(current_angles)
                    print(f"实际关节角度: {[f'{math.degrees(a):.2f}°' for a in current_angles]}")
                    print(f"当前末端位置: {P_current}")
                    print(f"当前末端姿态: {np.rad2deg(Phi_current)}°")
                    
                    # 计算初始误差
                    pos_error, angle_error = controller.calculate_pose_error(P_current, Phi_current)
                    print(f"\n📏 初始误差:")
                    print(f"位置误差: {pos_error*1000:.2f}mm")
                    print(f"姿态误差: {math.degrees(angle_error):.2f}°")
                else:
                    print(f"❌ 无法获取当前关节角度")
                    return
                
                # 显示目标和控制策略
                print(f"\n🎯 控制目标:")
                print(f"目标末端位置: {controller.P_target}")
                print(f"目标末端姿态: {controller.Phi_target_deg}°")
                print(f"位置精度要求: {controller.pos_tolerance*1000:.1f}mm")
                print(f"姿态精度要求: {math.degrees(controller.angle_tolerance):.1f}°")
                
                print(f"\n🔄 控制策略:")
                print(f"1. 实时获取机械臂实际关节角度")
                print(f"2. 调用planning(P_target, Phi_target, theta_actual)获得角速度")
                print(f"3. 积分计算: theta_target = theta_actual + theta_dot * dt")
                print(f"4. 发送theta_target命令到机械臂")
                print(f"5. 重复直到精度达到")
                print(f"控制频率: {1/controller.dt:.0f}Hz")
                
                print(f"\n" + "⚠️"*20)
                print(f"🚨 注意事项:")
                print(f"   • 真正的实时闭环控制")
                print(f"   • 持续运行直到达到精度要求")
                print(f"   • 无时间和迭代限制")
                print(f"   • 按Ctrl+C随时安全停止")
                print(f"   • 确保机械臂周围环境安全")
                print(f"⚠️"*20)
                
                # 用户确认
                input(f"\n🚀 按Enter键启动在线控制（每0.5s更新并发送角度），或Ctrl+C退出...")

                # 在线控制：每0.2秒计算并发送一次新的期望角度
                controller.run_online_iterative_control(interval_s=0.5, inner_time_ratio=0.7, max_total_seconds=600, print_interval=5)
                
                print(f"\n感谢使用Kinova在线迭代规划控制！")
                
            except KeyboardInterrupt:
                print(f"\n⚠️ 用户中断程序")
                if 'controller' in locals():
                    controller.emergency_stop()
            except Exception as e:
                print(f"\n❌ 程序异常: {e}")
                import traceback
                traceback.print_exc()
                if 'controller' in locals():
                    controller.emergency_stop()

if __name__ == "__main__":
    main()