import numpy as np

from Kinematic_fcn import Ttrans7

def inverse_kinematics(theta, V_edp, z_tool):
    """
    计算关节角速度（改进版，基于对 J 的 SVD 的阻尼伪逆）
    
    参数:
      theta: (7,) 当前关节角度
      V_edp: (6,) 末端期望速度 [vx,vy,vz,wx,wy,wz]
      z_tool: 工具长度
      
    返回:
      theta_dot: (7,) 关节角速度
      s: (6,) J_em 的奇异值（用于诊断奇异性）
    
    改进要点:
    1. 直接对 J_em (6x7) 做 SVD，得到正确的奇异值 s
    2. 使用标准的阻尼最小二乘公式: s_i/(s_i^2 + lambda^2)
    3. 自适应阻尼: 根据最小奇异值平滑调整阻尼系数
    4. 返回真实的雅可比奇异值，便于监控和调参
    """
    # 获取变换矩阵
    T_trans_0, T_0_7 = Ttrans7(theta)
    
    # 构造工具坐标系变换矩阵 T_7_e
    T_7_e = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1, z_tool],
        [0,  0,  0,  1]
    ], dtype=float)
    
    # 计算末端执行器在基坐标系中的变换
    T_0_e = T_0_7 @ T_7_e
    P_e = T_0_e[:3, 3].reshape(3,)  # 末端位置 (3,)
    
    # 初始化雅可比矩阵
    J_ev = np.zeros((3, 7))  # 线速度雅可比
    J_ew = np.zeros((3, 7))  # 角速度雅可比
    
    z_0 = np.array([0., 0., 1.])  # 基坐标系 z 轴
    
    # 计算雅可比矩阵每一列
    for i in range(7):
        # 获取关节 i 的变换矩阵
        T_0_i = T_trans_0[i, :, :]
        R_i = T_0_i[:3, :3]
        z = R_i @ z_0  # 关节轴方向（世界坐标系）
        
        # 关节位置
        p_tilde = T_0_i[:3, 3]
        p_ei_1 = P_e - p_tilde  # 从关节到末端的向量
        
        # 线速度雅可比: J_v = z × (p_e - p_i)
        J_ev[:, i] = np.cross(z, p_ei_1)
        
        # 角速度雅可比: J_w = z
        J_ew[:, i] = z
    
    # 组合完整雅可比矩阵 (6x7)
    J_em = np.vstack([J_ev, J_ew])
    
    # ========================================
    # 关键改进：直接对 J_em 做 SVD
    # ========================================
    # U: 6x6, s: 6个奇异值（降序）, Vt: 6x7
    U, s, Vt = np.linalg.svd(J_em, full_matrices=False)
    V = Vt.T  # 7x6
    
    # ========================================
    # 自适应阻尼：根据最小奇异值决定阻尼系数
    # ========================================
    lambda0=0.01
    eps=0.001
    s_min = np.min(s)
    if s_min < eps:
        # 当最小奇异值小于阈值时，平滑增加阻尼
        # lambda_val 的大小会随 s_min 减小而增大
        lambda_val = lambda0 * (1.0 - (s_min / eps)**2)
    else:
        # 远离奇异配置时不引入阻尼
        lambda_val = 0.0
    
    lambda2 = lambda_val**2
    
    # ========================================
    # 构建阻尼伪逆: J_pinv = V * S_damped_inv * U^T
    # ========================================
    # S_damped_inv 是 6x6 对角矩阵，对角元素为 s_i/(s_i^2 + lambda^2)
    S_inv = np.zeros((V.shape[1], U.shape[1]))  # 6x6
    for i in range(len(s)):
        S_inv[i, i] = s[i] / (s[i]**2 + lambda2)
    
    # 阻尼伪逆矩阵 (7x6)
    J_pinv = V @ S_inv @ U.T
    
    # ========================================
    # 主任务解：theta_dot = J_pinv * V_edp
    # ========================================
    theta_dot = J_pinv @ V_edp.reshape(6,)
    
    
    # 返回关节速度和奇异值（用于监控）
    return theta_dot, s