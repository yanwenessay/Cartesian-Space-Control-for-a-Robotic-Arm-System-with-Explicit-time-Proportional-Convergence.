"""
Robotic Arm Kinematic Model.

Provides forward kinematics, geometric Jacobian, and pose-error utilities
for an n-DOF serial manipulator described by standard DH parameters.

Reference:
    Cartesian Space Control and Joint Tracking Control for a Robotic Arm
    System with Explicit-time Proportional Convergence.
    IEEE/CAA Journal of Automatica Sinica, 2026.
    doi: 10.1109/JAS.2026.125963
"""

import numpy as np


class RobotModel:
    """
    n-DOF serial robotic arm described by Denavit-Hartenberg (DH) parameters.

    Each joint uses the standard DH convention::

        T_i = Rot_z(theta_i) * Trans_z(d_i) * Trans_x(a_i) * Rot_x(alpha_i)

    Parameters
    ----------
    dh_params : list of dict
        One entry per joint.  Each dict must contain:

        * ``'a'``            – link length [m]
        * ``'d'``            – link offset [m]
        * ``'alpha'``        – link twist [rad]
        * ``'theta_offset'`` – constant joint-angle offset [rad] (default 0)
    """

    def __init__(self, dh_params):
        if not dh_params:
            raise ValueError("dh_params must contain at least one joint.")
        self.dh_params = dh_params
        self.n_joints = len(dh_params)

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dh_transform(a, d, alpha, theta):
        """Return the 4x4 DH homogeneous transformation matrix."""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,   sa,       ca,      d     ],
            [0,   0,        0,       1     ],
        ])

    def _joint_transforms(self, theta):
        """
        Compute the cumulative homogeneous transforms from base to each frame.

        Returns a list ``T[0..n]`` where ``T[0]`` is the identity (base frame)
        and ``T[i]`` is the transform from base to joint *i* frame.
        """
        transforms = [np.eye(4)]
        for i, params in enumerate(self.dh_params):
            theta_i = theta[i] + params.get("theta_offset", 0.0)
            T_i = self._dh_transform(
                params["a"], params["d"], params["alpha"], theta_i
            )
            transforms.append(transforms[i] @ T_i)
        return transforms

    # ------------------------------------------------------------------
    # Public kinematics API
    # ------------------------------------------------------------------

    def forward_kinematics(self, theta):
        """
        Compute the end-effector homogeneous transformation matrix.

        Parameters
        ----------
        theta : array-like, shape (n_joints,)
            Joint angles [rad].

        Returns
        -------
        T : ndarray, shape (4, 4)
            End-effector transformation matrix with respect to the base frame.
        """
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n_joints,):
            raise ValueError(
                f"Expected theta with {self.n_joints} elements, got {theta.shape}."
            )
        transforms = self._joint_transforms(theta)
        return transforms[-1]

    def jacobian(self, theta):
        """
        Compute the geometric Jacobian J_m ∈ ℝ^{6×n}.

        The Jacobian maps joint velocities to end-effector spatial velocity:
        ``V_e = J_m @ theta_dot``

        Columns are built with the standard cross-product formula for
        revolute joints::

            J_linear_i   = z_{i-1}  ×  (p_e − p_{i-1})
            J_angular_i  = z_{i-1}

        Parameters
        ----------
        theta : array-like, shape (n_joints,)
            Joint angles [rad].

        Returns
        -------
        J : ndarray, shape (6, n_joints)
            Geometric Jacobian.
        """
        theta = np.asarray(theta, dtype=float)
        transforms = self._joint_transforms(theta)

        p_e = transforms[-1][:3, 3]  # end-effector position
        J = np.zeros((6, self.n_joints))
        for i in range(self.n_joints):
            z_i = transforms[i][:3, 2]  # z-axis of frame i
            p_i = transforms[i][:3, 3]  # origin of frame i
            J[:3, i] = np.cross(z_i, p_e - p_i)
            J[3:, i] = z_i
        return J

    def pose_error(self, T_current, T_desired):
        """
        Compute the 6-D pose error ``ε_e = [position_error; orientation_error]``.

        Follows the paper's sign convention: ``ε_e = current − desired``, so
        that the velocity-level dynamics satisfy::

            ε̇_e = J_m θ̇_p + ΔV − V_ed          (Eq. 6)

        A zero error vector means the current pose coincides with the desired
        pose.  A positive component means the current pose *exceeds* the
        desired pose along that dimension.

        Position error is the Euclidean difference ``p_current − p_desired``.
        Orientation error uses the axis-angle vector derived from the relative
        rotation ``R_err = R_current @ R_desired.T``.

        Parameters
        ----------
        T_current : ndarray, shape (4, 4)
            Current end-effector transformation.
        T_desired : ndarray, shape (4, 4)
            Desired end-effector transformation.

        Returns
        -------
        epsilon_e : ndarray, shape (6,)
            Pose error ``[Δp (3,); Δo (3,)]`` (current minus desired).
        """
        pos_error = T_current[:3, 3] - T_desired[:3, 3]

        R_cur = T_current[:3, :3]
        R_des = T_desired[:3, :3]
        R_err = R_cur @ R_des.T

        # Axis-angle from rotation matrix
        trace_val = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        angle = np.arccos(trace_val)
        if abs(angle) < 1e-9:
            ori_error = np.zeros(3)
        else:
            ori_error = (
                np.array([
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1],
                ])
                / (2.0 * np.sin(angle))
                * angle
            )

        return np.concatenate([pos_error, ori_error])

    def end_effector_velocity(self, theta, theta_dot):
        """
        Compute end-effector spatial velocity ``V_e = J_m @ theta_dot``.

        Parameters
        ----------
        theta : array-like, shape (n_joints,)
            Current joint angles [rad].
        theta_dot : array-like, shape (n_joints,)
            Current joint angular velocities [rad/s].

        Returns
        -------
        V_e : ndarray, shape (6,)
            End-effector spatial velocity ``[v (3,); ω (3,)]``.
        """
        J = self.jacobian(np.asarray(theta, dtype=float))
        return J @ np.asarray(theta_dot, dtype=float)


# ---------------------------------------------------------------------------
# Convenience factory – pre-built 6-DOF UR-style robot
# ---------------------------------------------------------------------------

def build_ur5_model():
    """
    Return a :class:`RobotModel` with approximate UR5 DH parameters.

    The values follow the modified DH convention used in the UR5 data-sheet
    (units: metres / radians).

    Returns
    -------
    robot : RobotModel
        Six-DOF model suitable for demonstration and testing.
    """
    dh_params = [
        {"a": 0.0,      "d": 0.089159, "alpha": np.pi / 2, "theta_offset": 0.0},
        {"a": -0.42500, "d": 0.0,      "alpha": 0.0,        "theta_offset": 0.0},
        {"a": -0.39225, "d": 0.0,      "alpha": 0.0,        "theta_offset": 0.0},
        {"a": 0.0,      "d": 0.10915,  "alpha": np.pi / 2,  "theta_offset": 0.0},
        {"a": 0.0,      "d": 0.09465,  "alpha": -np.pi / 2, "theta_offset": 0.0},
        {"a": 0.0,      "d": 0.0823,   "alpha": 0.0,        "theta_offset": 0.0},
    ]
    return RobotModel(dh_params)
