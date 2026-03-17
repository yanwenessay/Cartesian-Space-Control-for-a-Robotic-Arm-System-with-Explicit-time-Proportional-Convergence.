"""
Pose-level Indirect Cartesian Space Controller (Lyapunov-based).

Implements the control model from:

    Cartesian Space Control and Joint Tracking Control for a Robotic Arm
    System with Explicit-time Proportional Convergence.
    IEEE/CAA Journal of Automatica Sinica, 2026.
    doi: 10.1109/JAS.2026.125963

Core equation (Eq. 6 of the paper)
------------------------------------
The velocity-level pose error satisfies::

    ε̇_e = J_m θ̇_p + ΔV − V_ed          (6)

where

* ``ε̇_e``     – time derivative of the end-effector pose error
* ``J_m``     – Jacobian matrix evaluated at the current joint configuration
* ``θ̇_p``    – planned joint angular velocity (the **control input**)
* ``ΔV``      – end-effector spatial velocity tracking error, ``V_e − V_ep``
* ``V_ep``    – planned end-effector spatial velocity
* ``V_ed``    – desired end-effector spatial velocity

Lyapunov-based control law
---------------------------
Choose the quadratic Lyapunov candidate::

    𝒱 = ½ εₑᵀ εₑ

Its time derivative along system trajectories is::

    𝒱̇ = εₑᵀ ε̇ₑ = εₑᵀ (J_m θ̇_p + ΔV − V_ed)

Design the control input so that ``ε̇ₑ = −K(t) εₑ``, which gives
``𝒱̇ = −εₑᵀ K(t) εₑ ≤ 0`` (negative semi-definite for K ≻ 0), guaranteeing
asymptotic convergence of the pose error to zero.  Solving for the control
input yields::

    θ̇_p = J_m⁺ (V_ed − ΔV − K(t) εₑ)

where ``J_m⁺`` is the damped pseudoinverse of the Jacobian.

Explicit-time proportional convergence
---------------------------------------
With a constant gain ``K₀ > 0`` and a desired convergence time ``T_f``, the
time-varying gain::

    K(t) = K₀ / (T_f − t),   t ∈ [0, T_f)

produces the closed-form solution::

    εₑ(t) = εₑ(0) · ((T_f − t) / T_f)^{K₀}

so ``εₑ(T_f) = 0`` exactly (explicit-time convergence).  For ``K₀ = 1``
this simplifies to linear-in-time decay (proportional convergence).
"""

import numpy as np


class CartesianSpaceController:
    """
    Lyapunov-based pose-level indirect Cartesian space controller.

    Parameters
    ----------
    robot_model : robot_model.RobotModel
        Kinematic model of the robotic arm (used to compute ``J_m``).
    K : float or array-like, shape (6,) or (6, 6)
        Base control gain.  A scalar is expanded to ``K · I₆``.
        A 1-D array of length 6 is treated as a diagonal matrix.
    T_f : float or None, optional
        Desired convergence time [s] for explicit-time proportional
        convergence.  ``None`` (default) disables explicit-time mode and
        uses the constant gain ``K`` directly (exponential convergence).
    damping : float, optional
        Tikhonov damping coefficient ``λ`` for the damped pseudoinverse
        ``J⁺ = Jᵀ (J Jᵀ + λ² I)⁻¹``.  Default ``1e-3``.
    """

    def __init__(self, robot_model, K, T_f=None, damping=1e-3):
        self.robot = robot_model
        self.T_f = T_f
        self.damping = float(damping)

        # Normalise gain to a (6, 6) matrix
        K = np.asarray(K, dtype=float)
        if K.ndim == 0:                         # scalar
            self._K0 = float(K) * np.eye(6)
        elif K.ndim == 1 and K.shape == (6,):   # diagonal vector
            self._K0 = np.diag(K)
        elif K.ndim == 2 and K.shape == (6, 6): # full matrix
            self._K0 = K.copy()
        else:
            raise ValueError(
                "K must be a scalar, a 1-D array of length 6, "
                "or a 2-D (6×6) matrix."
            )

    # ------------------------------------------------------------------
    # Time-varying gain
    # ------------------------------------------------------------------

    def compute_gain(self, t):
        """
        Return the gain matrix ``K(t)`` at time *t*.

        In explicit-time mode (``T_f`` is set)::

            K(t) = K₀ / (T_f − t),   t < T_f
            K(t) = K₀ · 1e6,          t ≥ T_f  (saturation – hold at zero)

        Otherwise the constant base gain ``K₀`` is returned.

        Parameters
        ----------
        t : float
            Current simulation time [s].

        Returns
        -------
        K_t : ndarray, shape (6, 6)
        """
        if self.T_f is None:
            return self._K0.copy()

        if t >= self.T_f:
            # After the target convergence time: very large gain maintains
            # the zero-error state.
            return 1.0e6 * self._K0

        time_factor = 1.0 / (self.T_f - t)
        return self._K0 * time_factor

    # ------------------------------------------------------------------
    # Pseudoinverse
    # ------------------------------------------------------------------

    def pseudoinverse(self, J):
        """
        Compute the damped pseudoinverse of *J*.

        Uses Tikhonov regularisation::

            J⁺ = Jᵀ (J Jᵀ + λ² I_m)⁻¹

        Parameters
        ----------
        J : ndarray, shape (m, n)
            Matrix to invert.

        Returns
        -------
        J_pinv : ndarray, shape (n, m)
        """
        m = J.shape[0]
        lam2 = self.damping ** 2
        A = J @ J.T + lam2 * np.eye(m)
        return J.T @ np.linalg.inv(A)

    # ------------------------------------------------------------------
    # Core dynamics (Equation 6)
    # ------------------------------------------------------------------

    def pose_error_dynamics(self, J_m, theta_dot_p, delta_V, V_ed):
        """
        Evaluate the velocity-level pose error dynamics (Equation 6)::

            ε̇_e = J_m θ̇_p + ΔV − V_ed

        Parameters
        ----------
        J_m : ndarray, shape (6, n)
            Jacobian matrix.
        theta_dot_p : ndarray, shape (n,)
            Planned joint angular velocities [rad/s].
        delta_V : ndarray, shape (6,)
            End-effector velocity tracking error ``V_e − V_ep``.
        V_ed : ndarray, shape (6,)
            Desired end-effector spatial velocity.

        Returns
        -------
        epsilon_dot_e : ndarray, shape (6,)
            Time derivative of the pose error.
        """
        return J_m @ theta_dot_p + delta_V - V_ed

    # ------------------------------------------------------------------
    # Control law
    # ------------------------------------------------------------------

    def compute_joint_velocity(
        self, theta, epsilon_e, V_ed, delta_V=None, t=0.0
    ):
        """
        Compute the planned joint angular velocities ``θ̇_p``.

        Lyapunov-based control law::

            θ̇_p = J_m⁺ (V_ed − ΔV − K(t) εₑ)

        This drives ``ε̇ₑ = −K(t) εₑ``, making ``𝒱̇ ≤ 0`` and guaranteeing
        convergence of the pose error.

        Parameters
        ----------
        theta : array-like, shape (n_joints,)
            Current joint angles [rad].
        epsilon_e : array-like, shape (6,)
            Current pose error.
        V_ed : array-like, shape (6,)
            Desired end-effector spatial velocity.
        delta_V : array-like, shape (6,) or None
            End-effector spatial velocity tracking error ``V_e − V_ep``.
            Defaults to **zero** (perfect velocity tracking assumption).
        t : float, optional
            Current simulation time [s].  Used for explicit-time gain.

        Returns
        -------
        theta_dot_p : ndarray, shape (n_joints,)
            Planned joint angular velocities [rad/s].
        J_m : ndarray, shape (6, n_joints)
            Jacobian used in this computation (returned for diagnostics).
        """
        theta = np.asarray(theta, dtype=float)
        epsilon_e = np.asarray(epsilon_e, dtype=float)
        V_ed = np.asarray(V_ed, dtype=float)
        delta_V = (
            np.zeros(6) if delta_V is None else np.asarray(delta_V, dtype=float)
        )

        J_m = self.robot.jacobian(theta)
        J_pinv = self.pseudoinverse(J_m)
        K_t = self.compute_gain(t)

        control_signal = V_ed - delta_V - K_t @ epsilon_e
        theta_dot_p = J_pinv @ control_signal
        return theta_dot_p, J_m

    # ------------------------------------------------------------------
    # Stability diagnostics
    # ------------------------------------------------------------------

    def lyapunov_value(self, epsilon_e):
        """
        Compute the Lyapunov function value ``𝒱 = ½ ‖εₑ‖²``.

        Parameters
        ----------
        epsilon_e : array-like, shape (6,)

        Returns
        -------
        V : float
        """
        e = np.asarray(epsilon_e, dtype=float)
        return 0.5 * float(e @ e)

    def lyapunov_derivative(
        self, epsilon_e, theta_dot_p, J_m, delta_V, V_ed
    ):
        """
        Compute ``𝒱̇ = εₑᵀ ε̇ₑ`` for stability monitoring.

        A negative value confirms that the Lyapunov function is decreasing
        (the system is converging).

        Parameters
        ----------
        epsilon_e : array-like, shape (6,)
        theta_dot_p : array-like, shape (n_joints,)
        J_m : ndarray, shape (6, n_joints)
        delta_V : array-like, shape (6,)
        V_ed : array-like, shape (6,)

        Returns
        -------
        V_dot : float
            Should be ≤ 0 for a stable controller.
        """
        e = np.asarray(epsilon_e, dtype=float)
        e_dot = self.pose_error_dynamics(
            J_m,
            np.asarray(theta_dot_p, dtype=float),
            np.asarray(delta_V, dtype=float),
            np.asarray(V_ed, dtype=float),
        )
        return float(e @ e_dot)
