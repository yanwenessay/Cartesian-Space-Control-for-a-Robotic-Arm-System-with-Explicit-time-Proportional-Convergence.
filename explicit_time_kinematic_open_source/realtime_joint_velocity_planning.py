import numpy as np
import math
from planning_main import planning  
from Kinematic_fcn import Kinematic

def realtime_joint_velocity_planning(theta_actual, P_target, Phi_target, 
                                   z_tool=-0.16746, max_joint_velocity=0.5, 
                                   velocity_scale_factor=0.8):
    """
    实时关节角速度规划函数
    
    基于当前实际关节角度和目标位姿，计算实时关节角速度
    
    Args:
        theta_actual (numpy.array): 当前实际关节角度 (弧度) [7,]
        P_target (numpy.array): 目标末端位置 [x, y, z] (米) [3,]
        Phi_target (numpy.array): 目标末端姿态角 (弧度) [3,]
        z_tool (float): 工具长度 (米)，默认 -0.16746
        max_joint_velocity (float): 最大关节角速度限制 (rad/s)，默认 0.5
        velocity_scale_factor (float): 速度缩放因子，默认 0.8
        
    Returns:
        numpy.array: 实时规划的关节角速度 (rad/s) [7,]
        
    Raises:
        ValueError: 当输入参数不合法时
        Exception: 当规划计算失败时
    """
    
    # 输入验证
    if not isinstance(theta_actual, np.ndarray) or theta_actual.shape != (7,):
        raise ValueError("theta_actual必须是形状为(7,)的numpy数组")
    
    if not isinstance(P_target, np.ndarray) or P_target.shape != (3,):
        raise ValueError("P_target必须是形状为(3,)的numpy数组")
        
    if not isinstance(Phi_target, np.ndarray) or Phi_target.shape != (3,):
        raise ValueError("Phi_target必须是形状为(3,)的numpy数组")
    
    if max_joint_velocity <= 0:
        raise ValueError("max_joint_velocity必须大于0")
        
    if not (0 < velocity_scale_factor <= 1):
        raise ValueError("velocity_scale_factor必须在(0,1]范围内")
    
    try:
        # 调用planning函数计算关节角速度
        theta_dot_planning = planning(P_target, Phi_target, theta_actual)
        
        # 转换为numpy数组
        theta_dot_planning = np.array(theta_dot_planning)
        
        # 验证输出维度
        if theta_dot_planning.shape != (7,):
            raise ValueError(f"planning函数返回维度错误: {theta_dot_planning.shape}, 期望(7,)")
        
        # 限制最大角速度
        theta_dot_limited = np.clip(theta_dot_planning, 
                                  -max_joint_velocity, 
                                  max_joint_velocity)
        
        # 应用速度缩放因子
        theta_dot_scaled = theta_dot_limited * velocity_scale_factor
        
        return theta_dot_scaled
        
    except Exception as e:
        raise Exception(f"实时关节角速度规划失败: {e}")


def get_current_pose_from_joints(theta_actual, z_tool=-0.16746):
    """
    从关节角度计算当前末端位姿
    
    Args:
        theta_actual (numpy.array): 当前关节角度 (弧度) [7,]
        z_tool (float): 工具长度 (米)
        
    Returns:
        tuple: (位置[3,], 姿态[3,], 成功标志)
    """
    try:
        P_current, Phi_current, _, _ = Kinematic(theta_actual, z_tool)
        return P_current, Phi_current, True
    except Exception as e:
        print(f"正运动学计算失败: {e}")
        return np.zeros(3), np.zeros(3), False


def calculate_pose_error(P_current, Phi_current, P_target, Phi_target):
    """
    计算位姿误差
    
    Args:
        P_current (numpy.array): 当前位置 [3,]
        Phi_current (numpy.array): 当前姿态 [3,]
        P_target (numpy.array): 目标位置 [3,]
        Phi_target (numpy.array): 目标姿态 [3,]
        
    Returns:
        tuple: (位置误差, 姿态误差)
    """
    pos_error = np.linalg.norm(P_current - P_target)
    angle_error = np.linalg.norm(Phi_current - Phi_target)
    return pos_error, angle_error

