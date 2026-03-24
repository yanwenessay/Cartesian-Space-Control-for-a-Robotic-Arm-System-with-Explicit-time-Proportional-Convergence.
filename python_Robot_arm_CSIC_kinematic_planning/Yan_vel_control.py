import numpy as np
from scipy.spatial.transform import Rotation as R
from Kinematic_fcn import Kinematic
import constant as cont

def vel_control(V_ed, theta, P_ed, Phi_ed): #return V_edp, p_error, delta_error

    P_e, Phi_e, quater_t, T_0_e = Kinematic(theta, cont.z_tool) #调用正运动学程序

    R_et = T_0_e[:3, :3]   # 实时末端旋转矩阵
    P_et = T_0_e[:3, 3]    # 实时末端位置
    
    # 期望姿态转为旋转矩阵 (Z-Y-X欧拉角)
    R_ed = R.from_euler('ZYX', Phi_ed, degrees=False).as_matrix()
    
    # 计算姿态误差（旋转矩阵差）
    R_t_d = R_et @ R_ed.T  # 实时时刻相对期望时刻的旋转矩阵
    
    # 将旋转矩阵转为轴角表示
    try:
        r = R.from_matrix(R_t_d)
        rotvec = r.as_rotvec()
        phi_ee = np.linalg.norm(rotvec)
        k_ee = rotvec / phi_ee if phi_ee > 1e-6 else np.zeros(3)
    except:
        phi_ee = 0.0
        k_ee = np.zeros(3)
    
    # 计算位姿误差
    p_error = P_et - P_ed
    delta_error = phi_ee * k_ee  # 旋转误差向量
    
    # 轨迹规划参数
    Tc = 2.0
    xc1 = 500 / 1000      # 500 mm
    xc2 = 40 * np.pi / 180  # 40度转弧度
    xs1 = 0.0001          # 0.1 mm
    xs2 = 0.1 * np.pi / 180  # 0.1度转弧度
    kc1 = 1.0
    kc2 = 1.0
    
    # 计算补偿速度
    Ved_comp = np.zeros(6)
    Ved_comp[:3] = -kc1 * np.log(xc1 / xs1) / Tc * p_error
    Ved_comp[3:] = -kc2 * np.log(xc2 / xs2) / Tc * delta_error
    
    # 总速度
    V_edp = np.zeros(6)
    V_edp[:3]=V_ed[:3]+Ved_comp[:3]
    V_edp[3:]=V_ed[3:]+Ved_comp[3:]

    return V_edp, p_error, delta_error

