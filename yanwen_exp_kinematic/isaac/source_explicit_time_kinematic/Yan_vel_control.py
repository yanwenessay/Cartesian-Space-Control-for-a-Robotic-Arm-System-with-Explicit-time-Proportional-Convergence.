import numpy as np
from scipy.spatial.transform import Rotation as R
from Kinematic_fcn import Kinematic
import constant as cont


def skew(a):
    a = np.asarray(a, dtype=float).reshape(3)
    return np.array([
        [0.0, -a[2], a[1]],
        [a[2], 0.0, -a[0]],
        [-a[1], a[0], 0.0],
    ], dtype=float)


def left_jacobian_inv_so3(phi):
    """Inverse left Jacobian on SO(3) for analytical pose-error coordinates."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    angle = float(np.linalg.norm(phi))
    phi_x = skew(phi)
    if angle < 1e-6:
        return np.eye(3) - 0.5 * phi_x + (1.0 / 12.0) * (phi_x @ phi_x)
    gamma = (1.0 / (angle * angle)
             - (1.0 + np.cos(angle)) / (2.0 * angle * np.sin(angle)))
    return np.eye(3) - 0.5 * phi_x + gamma * (phi_x @ phi_x)


def vel_control(V_ed, theta, P_ed, Phi_ed):
    """Analytical pose-feedback velocity command.

    Attitude error follows the updated paper formulation:
        R_err = R_e R_ed^T, phi = log(R_err)
        phi_dot = J_l^{-1}(phi) (omega_e - R_err omega_ed)
    The returned Lambda_e makes IK use Ja = Lambda_e Jm.
    """
    V_ed = np.asarray(V_ed, dtype=float).reshape(6)
    P_ed = np.asarray(P_ed, dtype=float).reshape(3)
    Phi_ed = np.asarray(Phi_ed, dtype=float).reshape(3)

    _, _, _, T_0_e = Kinematic(theta, cont.z_tool)
    R_e = T_0_e[:3, :3]
    P_e = T_0_e[:3, 3]
    R_ed = R.from_euler('ZYX', Phi_ed, degrees=False).as_matrix()

    R_err = R_e @ R_ed.T
    delta_error = R.from_matrix(R_err).as_rotvec()
    Jl_inv = left_jacobian_inv_so3(delta_error)
    p_error = P_e - P_ed

    Tc = 1.0
    xc1 = 2000 / 1000
    xc2 = 150 * np.pi / 180
    xs1 = 0.0001
    xs2 = 0.1 * np.pi / 180
    kc1 = 1.0
    kc2 = 1.0
    k_pos = kc1 * np.log(xc1 / xs1) / Tc
    k_rot = kc2 * np.log(xc2 / xs2) / Tc

    v_ed = V_ed[:3]
    omega_ed = V_ed[3:]
    V_ed_a = np.concatenate((v_ed, Jl_inv @ (R_err @ omega_ed)))
    eps = np.concatenate((p_error, delta_error))
    K_pose = np.diag([k_pos, k_pos, k_pos, k_rot, k_rot, k_rot])
    V_edp = V_ed_a - K_pose @ eps

    Lambda_e = np.eye(6)
    Lambda_e[3:, 3:] = Jl_inv
    return V_edp, p_error, delta_error, Lambda_e
