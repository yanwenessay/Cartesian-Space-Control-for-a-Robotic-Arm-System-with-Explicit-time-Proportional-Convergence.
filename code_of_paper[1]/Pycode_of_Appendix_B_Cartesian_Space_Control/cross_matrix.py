def cross_matrix(z, p):
    """
    不使用外部库实现叉乘矩阵运算
    
    参数:
    z: 3元素列表 [z1, z2, z3]
    p: 3元素列表 [p1, p2, p3]
    
    返回:
    y: 3元素列表 = Z*p
    """
    # 解构向量元素
    z1, z2, z3 = z
    p1, p2, p3 = p
    
    # 创建反对称矩阵 Z (不实际存储整个矩阵)
    # 直接计算结果向量 y = Z * p
    y = [
        0*p1 + (-z3)*p2 + z2*p3,  # 第一行
        z3*p1 + 0*p2 + (-z1)*p3,  # 第二行
        (-z2)*p1 + z1*p2 + 0*p3   # 第三行
    ]
    
    return y

# 示例用法
#z = [1, 2, 3]
#p = [4, 5, 6]
#result = cross_matrix(z, p)
#print("结果:", result)  # 输出: [(-3)*5 + 2 * 6, 3 * 4 + (-1)*6, (-2)*4 + 1 * 5]

#import numpy as np

#a = np.array([1,2,3])
#b = np.array([4,5,6])
#c = np.cross(a,b)
#print("test",c)