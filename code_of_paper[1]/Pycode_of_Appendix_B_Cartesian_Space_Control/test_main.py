import numpy as np
import matplotlib.pyplot as plt
from planning_main import planning  
from Kinematic_fcn import Kinematic


P_ed = np.array([-0.440, -0.015, 0.124])
Phi_ed_deg = np.array([-97.8, -79.5, -133.7])  # 已经是度数ZYX 欧拉角
def joint_angle_integration_planning(theta_init=None, dt=0.001, max_time=10.0,
                          tolerance=1e-6, max_iterations=1000):
    """关节角度积分控制函数 - 7自由度机械臂"""
   
    # 修复1: 角度转换问题
    Phi_ed = np.deg2rad(Phi_ed_deg)    # 转换为弧度
   
    # 修复2: theta应该从参数获取，不是从self.base_feedback
    if theta_init is None:
        theta = np.zeros(7)  # 7自由度机械臂
    else:
        theta = np.array(theta_init)
   
    # 修复3: 调用planning_main函数，使用当前关节角度theta作为实时关节角度
    theta_dot_planning = planning(P_ed, Phi_ed, theta)  # 使用theta作为当前实时关节角度
    
    # 修复4: 积分更新
    theta_planning = theta + np.array(theta_dot_planning) * dt
   
    return theta_planning

def calculate_pose_error(P_current, Phi_current, P_desired, Phi_desired):
    """计算位姿误差"""
    # 位置误差
    pos_error = np.linalg.norm(P_current - P_desired)
    
    # 姿态误差 (角度差)
    angle_error = np.linalg.norm(Phi_current - Phi_desired)
    
    return pos_error, angle_error

def joint_angle_integration_full(theta_init=None, dt=0.001, max_time=50.0,
                               pos_tolerance=1e-4, angle_tolerance=1e-4, 
                               max_iterations=100000):
    """完整的关节角度积分控制函数 - 基于末端位姿误差收敛"""
    
    # 期望位姿
    Phi_ed = np.deg2rad(Phi_ed_deg)
    
    z_tool = -0.16746  # 机械臂六轴到末端的向z轴的标量位置
    
    # 初始化
    if theta_init is None:
        theta = np.zeros(7)
    else:
        theta = np.array(theta_init)
    
    # 记录历史数据
    theta_history = [theta.copy()]
    time_history = [0.0]
    pos_error_history = []
    angle_error_history = []
    
    current_time = 0.0
    iteration = 0
    converged = False
    
    print("开始迭代积分...")
    print(f"期望位置: {P_ed}")
    print(f"期望姿态(度): {Phi_ed_deg}")
    print(f"期望姿态(弧度): {Phi_ed}")
    
    while current_time < max_time and iteration < max_iterations:
        # 计算当前末端位姿
        try:
            P_current, Phi_current, _, _ = Kinematic(theta, z_tool)
        except Exception as e:
            print(f"正运动学计算错误: {e}")
            break
            
        # 计算位姿误差
        pos_error, angle_error = calculate_pose_error(P_current, Phi_current, P_ed, Phi_ed)
        
        # 记录误差历史
        pos_error_history.append(pos_error)
        angle_error_history.append(angle_error)
        
        # 检查收敛 - 基于末端位姿误差
        if pos_error < pos_tolerance and angle_error < angle_tolerance:
            converged = True
            print(f"\n收敛! 迭代次数: {iteration}")
            print(f"最终位置误差: {pos_error:.6f} m")
            print(f"最终姿态误差: {angle_error:.6f} rad")
            break
        
        # 每1000次迭代打印一次状态
        if iteration % 1000 == 0:
            print(f"迭代 {iteration}: 位置误差={pos_error:.6f}m, 姿态误差={angle_error:.6f}rad")
            print(f"  当前位置: {P_current}")
            print(f"  当前姿态: {np.rad2deg(Phi_current)}")
            
        
        # 计算下一步角度
        try:
            theta_new = joint_angle_integration_planning(theta_init=theta, dt=dt)
        except Exception as e:
            print(f"规划计算错误: {e}")
            break
        
        # 更新状态
        theta = theta_new
        current_time += dt
        iteration += 1
        
        # 记录历史(每隔100步记录一次，避免数据过多)
        if iteration % 100 == 0:
            theta_history.append(theta.copy())
            time_history.append(current_time)
    
    # 添加最终状态
    theta_history.append(theta.copy())
    time_history.append(current_time)
    
    # 最终验证
    try:
        P_final, Phi_final, _, _ = Kinematic(theta, z_tool)
        final_pos_error, final_angle_error = calculate_pose_error(P_final, Phi_final, P_ed, Phi_ed)
        
        print(f"\n=== 最终结果 ===")
        print(f"总迭代次数: {iteration}")
        print(f"总时间: {current_time:.3f} s")
        print(f"收敛状态: {converged}")
        print(f"最终位置误差: {final_pos_error:.6f} m")
        print(f"最终姿态误差: {final_angle_error:.6f} rad")
        print(f"期望位置: {P_ed}")
        print(f"实际位置: {P_final}")
        print(f"期望姿态(度): {Phi_ed_deg}")
        print(f"实际姿态(度): {np.rad2deg(Phi_final)}")
        
    except Exception as e:
        print(f"最终验证错误: {e}")
        P_final, Phi_final = None, None
    
    return theta, np.array(theta_history), np.array(time_history), converged, pos_error_history, angle_error_history

def plot_joint_trajectories(theta_history, time_history, title="Joint Angle Trajectories"):
    """绘制关节角度轨迹图 - 7个关节"""
    plt.figure(figsize=(15, 10))
   
    num_joints = theta_history.shape[1]
    # 7个关节，3行3列布局
    for i in range(num_joints):
        plt.subplot(3, 3, i+1)
        plt.plot(time_history, theta_history[:, i])
        plt.xlabel('Time (s)')
        plt.ylabel(f'Joint {i+1} Angle (rad)')
        plt.title(f'Joint {i+1}')
        plt.grid(True)
   
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_error_history(pos_error_history, angle_error_history, dt=0.001):
    """绘制误差历史图"""
    time_steps = np.arange(len(pos_error_history)) * dt
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.semilogy(time_steps, pos_error_history)
    plt.xlabel('Time (s)')
    plt.ylabel('Position Error (m)')
    plt.title('Position Error History')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.semilogy(time_steps, angle_error_history)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle Error (rad)')
    plt.title('Angle Error History')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def example_usage():
    """使用示例"""
    # 测试单步积分
    print("=== 测试单步积分 ===")
    theta_init = [10, 0, 10, 10, 0, 10, 0]
   
    try:
        theta_new = joint_angle_integration_planning(theta_init=theta_init, dt=0.01)
        print(f"单步积分结果: {theta_new}")
       
        # 测试完整积分过程 - 使用更严格的收敛标准和更多迭代
        print("\n=== 测试完整积分过程 ===")
        result = joint_angle_integration_full(
            theta_init=theta_init,
            dt=0.001,                    # 时间步长
            max_time=100.0,              # 增加最大时间
            pos_tolerance=1e-5,          # 位置误差容限 (0.01mm)
            angle_tolerance=1e-5,        # 角度误差容限
            max_iterations=200000        # 增加最大迭代次数
        )
        
        theta_final, theta_history, time_history, converged, pos_error_history, angle_error_history = result
       
        # 绘制轨迹图
        if len(theta_history) > 1:
            plot_joint_trajectories(theta_history, time_history)
            plot_error_history(pos_error_history, angle_error_history)
           
    except Exception as e:
        print(f"执行出错: {e}")
        import traceback
        traceback.print_exc()

# 实时控制接口
def get_next_joint_angles(current_theta, dt=0.01):
    """
    实时控制接口 - 根据当前关节角度计算下一步角度
   
    参数:
    current_theta: 当前关节角度(7维数组)
    dt: 时间步长
   
    返回:
    next_theta: 下一步关节角度
    """
    try:
        return joint_angle_integration_planning(theta_init=current_theta, dt=dt)
    except Exception as e:
        print(f"计算下一步关节角度时出错: {e}")
        return current_theta  # 出错时返回当前角度

if __name__ == "__main__":
    example_usage()