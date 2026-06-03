import numpy as np
from scipy.spatial.transform import Rotation as R
from Kinematic_fcn import Kinematic
import constant as cont

def vel_plan(theta0, p_ef, Phi_ef, t): #return Ved

    P_e, Phi_e, quater_t, T_0_e = Kinematic(theta0, cont.z_tool) #调用正运动学程序
    
    R_e0 = T_0_e[:3, :3]   # 初始旋转矩阵
    p_e0 = T_0_e[:3, 3]    # 末端初始位置
    
    # 将目标欧拉角转换为旋转矩阵（ZYX顺序）
    R_ef = R.from_euler('ZYX', Phi_ef, degrees=False).as_matrix()
    
    # 线速度规划
    e_p = p_ef - p_e0
    ts = 0.5  # 加减速时间
    tf = 1.5  # 规划总时间
    ved = np.zeros(3)  # 线速度
    
    if t > 0.001 and t < ts:
        ved = e_p / (tf - ts) * (t / ts)
    elif t >= ts and t <= (tf - ts):
        ved = e_p / (tf - ts)
    elif t >= (tf - ts) and t <= tf:
        ved = e_p / (tf - ts) * (1 - (t + ts - tf) / ts)
    
    ## 角速度规划
    # 计算相对旋转矩阵（从初始姿态到目标姿态）
    R_f_i = R_ef @ R_e0.T
    
    # 将旋转矩阵转为轴角表示
    try:
        r = R.from_matrix(R_f_i)
        rotvec = r.as_rotvec()
        phi = np.linalg.norm(rotvec)
        kd = rotvec / phi if phi > 1e-6 else np.zeros(3)
    except:
        phi = 0.0
        kd = np.zeros(3)
    
    # 规划旋转角速度（与线速度相同的梯形模式）
    w_mag = 0.0
    if t > 0.001 and t < ts:
        w_mag = phi / (tf - ts) * (t / ts)
    elif t >= ts and t <= (tf - ts):
        w_mag = phi / (tf - ts)
    elif t >= (tf - ts) and t <= tf:
        w_mag = phi / (tf - ts) * (1 - (t + ts - tf) / ts)
    
    # 计算角速度向量
    wed = w_mag * kd
    
    # 组合线速度和角速度
    Ved = np.concatenate((ved, wed))

    return Ved
