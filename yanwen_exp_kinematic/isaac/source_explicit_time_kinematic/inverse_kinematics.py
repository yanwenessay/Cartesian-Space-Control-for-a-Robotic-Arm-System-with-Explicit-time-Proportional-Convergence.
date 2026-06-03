import numpy as np

from Kinematic_fcn import Ttrans7


def inverse_kinematics(theta, V_edp, z_tool, analytical_transform=None):
    """Compute joint velocity using damped least-squares IK.

    When analytical_transform is provided, use Ja = Lambda_e Jm. This keeps
    the external interface compatible while making the planner consistent with
    the analytical pose-error formulation in the updated paper.
    """
    T_trans_0, T_0_7 = Ttrans7(theta)

    T_7_e = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1, z_tool],
        [0,  0,  0,  1]
    ], dtype=float)

    T_0_e = T_0_7 @ T_7_e
    P_e = T_0_e[:3, 3].reshape(3,)

    J_ev = np.zeros((3, 7))
    J_ew = np.zeros((3, 7))
    z_0 = np.array([0., 0., 1.])

    for i in range(7):
        T_0_i = T_trans_0[i, :, :]
        z = T_0_i[:3, :3] @ z_0
        p_i = T_0_i[:3, 3]
        J_ev[:, i] = np.cross(z, P_e - p_i)
        J_ew[:, i] = z

    J_m = np.vstack([J_ev, J_ew])
    if analytical_transform is not None:
        Lambda_e = np.asarray(analytical_transform, dtype=float).reshape(6, 6)
        J_used = Lambda_e @ J_m
    else:
        J_used = J_m

    U, s, Vt = np.linalg.svd(J_used, full_matrices=False)
    V = Vt.T

    lambda0 = 0.01
    eps = 0.001
    s_min = np.min(s)
    lambda_val = lambda0 * (1.0 - (s_min / eps) ** 2) if s_min < eps else 0.0
    lambda2 = lambda_val ** 2

    S_inv = np.zeros((V.shape[1], U.shape[1]))
    for i in range(len(s)):
        S_inv[i, i] = s[i] / (s[i] ** 2 + lambda2)

    J_pinv = V @ S_inv @ U.T
    theta_dot = J_pinv @ np.asarray(V_edp, dtype=float).reshape(6,)
    return theta_dot, s
