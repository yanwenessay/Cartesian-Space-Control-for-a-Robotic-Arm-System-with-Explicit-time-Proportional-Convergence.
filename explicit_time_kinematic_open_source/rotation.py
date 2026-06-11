import numpy as np
#完整库 eulzyx的完整正运动学
z, y, x = [0.35*np.pi, -0.15*np.pi, 0.75*np.pi] #eulzyx
# 计算每个角度的三角函数值
cz = np.cos(z)
sz = np.sin(z)
cy = np.cos(y)
sy = np.sin(y)
cx = np.cos(x)
sx = np.sin(x)
    
# 计算旋转矩阵元素
# R = Rz * Ry * Rx
r00 = cz * cy
r01 = cz * sy * sx - sz * cx
r02 = cz * sy * cx + sz * sx

r10 = sz * cy
r11 = sz * sy * sx + cz * cx
r12 = sz * sy * cx - cz * sx

r20 = -sy
r21 = cy * sx
r22 = cy * cx

# 组装旋转矩阵
rot_matrix = np.array([
    [r00, r01, r02],
    [r10, r11, r12],
    [r20, r21, r22]
])
    
eulzyx=np.degrees([z, y, x])
