import numpy as np
from scipy.spatial.transform import Rotation as R
## 旋转矩阵
def Ttrans7(theta): # 输出 T_trans_0, T_0_7, 输入参数theta应为包含7个关节角度的数组
    """
    注意D-H坐标系的建立方法，坐标系i的原点位于公垂线a_i与关节轴i的交点处。x_i沿a_i的方向由关节轴i指向关节轴i+1
    a_i：沿着X_i轴，从Z_i移动到Z_i+1的距离，alpha_i：沿着X_i轴，从Z_i旋转到Z_i+1的角度，
    d_i：沿着Z_i轴，从X_i-1移动到X_i的距离，theta_i：沿着Z_i轴，从X_i-1旋转到X_i的角度。
    转换矩阵和旋转矩阵

    根据关节角度计算基座到末端的齐次变换矩阵T_0_7
    参数:
        theta: 包含7个关节角度的列表（弧度）
    返回:
        4x4齐次变换矩阵（numpy数组）
    """
    joints = 7 # 七轴
    # DH参数
    theta_offset = [0, 0, 0, 0, 0, 0, 0]
    a =  [0, 0, 0, 0, 0, 0, 0]
    alpha = [np.pi, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2]
    d = [-0.28481, -0.01176, -0.42076, -0.01276, -0.31436, 0, 0]
    
    T_0_7 = np.eye(4)  # 初始单位矩阵
    T_trans = np.zeros((joints, 4, 4))   # 每个关节的局部变换矩阵
    T_trans_0 = np.zeros((joints, 4, 4)) # 每个关节到基坐标系的变换

    for i in range(joints):
        # 计算当前关节角度（含偏移）
        theta_i = theta[i] + theta_offset[i]
        # 计算三角函数值
        c_t, s_t = np.cos(theta_i), np.sin(theta_i)
        c_a, s_a = np.cos(alpha[i]), np.sin(alpha[i])
        
        # 构造当前关节的DH变换矩阵（与MATLAB代码逻辑一致）
        T_trans_i = np.array([
            [c_t,         -s_t,            0,           a[i]],
            [s_t * c_a,    c_t * c_a,     -s_a,        -s_a * d[i]],
            [s_t * s_a,    c_t * s_a,      c_a,         c_a * d[i]],
            [0,            0,              0,           1]
        ])
        
        T_trans[i] = T_trans_i  # 存储局部变换
        T_0_7 = T_0_7 @ T_trans_i  # 更新累积变换
    
         # 存储当前关节到基坐标系的变换
        T_trans_0[i] = T_0_7.copy()
    
    return T_trans_0, T_0_7


## 正运动学
def Kinematic(theta, z_tool): # 输出 P_e, Phi_e, quater_t, T_0_e; z_tool为沿着7轴直线延长的末端z轴
   
    _, T_0_7 = Ttrans7(theta) 

    T_7_e = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  z_tool],
        [0,  0,  0,  1]
    ])

    T_0_e = T_0_7 @ T_7_e
    # 提取 3 维工具位置
    P_e = T_0_e[:3, 3]  # 提取第四列的前三行
    # 提取旋转矩阵
    R_0_e = T_0_e[:3, :3] 
    # 将旋转矩阵转换为 ZYX 欧拉角 (顺序: Z旋转->Y旋转->X旋转)
    # 注意：返回顺序是 [z_angle, y_angle, x_angle] (单位: 弧度)
    rot = R.from_matrix(R_0_e)
    Phi_e = rot.as_euler('ZYX', degrees=False)  # 返回形状为 (3,) 的数组
    quaternion = rot.as_quat() 
    quater_t = np.array([
        quaternion[3],  # w
        quaternion[0],  # x
        quaternion[1],  # y
        quaternion[2]   # z
        ])
    return P_e, Phi_e, quater_t, T_0_e
    