"""
Unit tests for lyapunov_controller.CartesianSpaceController.

Tests cover:
* Controller initialisation (gain formats)
* Time-varying gain computation
* Damped pseudoinverse
* Pose error dynamics (Equation 6)
* Control law (joint velocity computation)
* Lyapunov value and derivative (stability check)
* Closed-loop convergence via simulation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from robot_model import build_ur5_model, RobotModel
from lyapunov_controller import CartesianSpaceController


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ur5():
    return build_ur5_model()


@pytest.fixture
def ctrl_const(ur5):
    """Controller with constant gain (no explicit-time mode)."""
    return CartesianSpaceController(ur5, K=2.0, T_f=None, damping=1e-3)


@pytest.fixture
def ctrl_explicit(ur5):
    """Controller with explicit-time convergence (T_f = 5 s)."""
    return CartesianSpaceController(ur5, K=1.0, T_f=5.0, damping=1e-3)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_scalar_gain_becomes_6x6(self, ur5):
        ctrl = CartesianSpaceController(ur5, K=3.0)
        np.testing.assert_allclose(ctrl._K0, 3.0 * np.eye(6))

    def test_vector_gain_becomes_diagonal(self, ur5):
        k_vec = np.arange(1, 7, dtype=float)
        ctrl = CartesianSpaceController(ur5, K=k_vec)
        np.testing.assert_allclose(ctrl._K0, np.diag(k_vec))

    def test_matrix_gain_stored_correctly(self, ur5):
        K = np.eye(6) * 5.0
        ctrl = CartesianSpaceController(ur5, K=K)
        np.testing.assert_allclose(ctrl._K0, K)

    def test_invalid_gain_shape_raises(self, ur5):
        with pytest.raises(ValueError):
            CartesianSpaceController(ur5, K=np.ones((3, 3)))

    def test_T_f_stored(self, ctrl_explicit):
        assert ctrl_explicit.T_f == 5.0

    def test_no_T_f_default(self, ctrl_const):
        assert ctrl_const.T_f is None


# ---------------------------------------------------------------------------
# Gain computation
# ---------------------------------------------------------------------------

class TestComputeGain:
    def test_constant_mode_unchanged(self, ctrl_const):
        K_t = ctrl_const.compute_gain(t=2.0)
        np.testing.assert_allclose(K_t, ctrl_const._K0)

    def test_explicit_time_scales_correctly(self, ctrl_explicit):
        # At t=4 with T_f=5: K(t) = K0 / (5-4) = K0
        K_t = ctrl_explicit.compute_gain(t=4.0)
        expected = ctrl_explicit._K0 / (5.0 - 4.0)
        np.testing.assert_allclose(K_t, expected)

    def test_explicit_time_at_zero(self, ctrl_explicit):
        K_t = ctrl_explicit.compute_gain(t=0.0)
        expected = ctrl_explicit._K0 / 5.0
        np.testing.assert_allclose(K_t, expected)

    def test_explicit_time_past_T_f(self, ctrl_explicit):
        K_t = ctrl_explicit.compute_gain(t=6.0)
        # Should be very large (saturation)
        assert np.all(K_t >= ctrl_explicit._K0)


# ---------------------------------------------------------------------------
# Pseudoinverse
# ---------------------------------------------------------------------------

class TestPseudoinverse:
    def test_shape(self, ctrl_const, ur5):
        theta = np.zeros(6)
        J = ur5.jacobian(theta)
        J_pinv = ctrl_const.pseudoinverse(J)
        assert J_pinv.shape == (6, 6)

    def test_square_matrix_approx_inverse(self, ur5):
        """For a well-conditioned square matrix the damped pinv ≈ true inverse."""
        ctrl = CartesianSpaceController(ur5, K=1.0, damping=1e-6)
        theta = np.array([0.1, -0.5, 0.3, -0.2, 0.4, -0.1])
        J = ur5.jacobian(theta)
        J_pinv = ctrl.pseudoinverse(J)
        # J @ J_pinv should be close to identity
        np.testing.assert_allclose(J @ J_pinv, np.eye(6), atol=1e-4)

    def test_rectangular_matrix_shape(self, ur5):
        """4×6 Jacobian (fewer rows) → pinv should be 6×4."""
        ctrl = CartesianSpaceController(ur5, K=1.0)
        J_rect = ur5.jacobian(np.zeros(6))[:4, :]   # truncate to 4 rows
        J_pinv = ctrl.pseudoinverse(J_rect)
        assert J_pinv.shape == (6, 4)


# ---------------------------------------------------------------------------
# Pose error dynamics (Equation 6)
# ---------------------------------------------------------------------------

class TestPoseErrorDynamics:
    def test_zero_inputs_give_zero_dynamics(self, ctrl_const, ur5):
        J_m = ur5.jacobian(np.zeros(6))
        theta_dot_p = np.zeros(6)
        delta_V = np.zeros(6)
        V_ed = np.zeros(6)
        e_dot = ctrl_const.pose_error_dynamics(J_m, theta_dot_p, delta_V, V_ed)
        np.testing.assert_allclose(e_dot, np.zeros(6), atol=1e-12)

    def test_equation6_holds(self, ctrl_const, ur5):
        """ε̇_e = J_m θ̇_p + ΔV − V_ed  must hold component-wise."""
        theta = np.array([0.1, -0.5, 0.3, -0.2, 0.4, -0.1])
        J_m = ur5.jacobian(theta)
        theta_dot_p = np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.05])
        delta_V = np.array([0.01, -0.01, 0.02, 0.0, 0.0, 0.01])
        V_ed = np.array([0.1, 0.0, 0.05, 0.0, 0.0, 0.0])

        e_dot = ctrl_const.pose_error_dynamics(J_m, theta_dot_p, delta_V, V_ed)
        expected = J_m @ theta_dot_p + delta_V - V_ed
        np.testing.assert_allclose(e_dot, expected, atol=1e-12)

    def test_only_V_ed_shifts_dynamics(self, ctrl_const, ur5):
        """
        With θ̇_p=0 and ΔV=0, Eq. 6 reduces to ε̇_e = −V_ed.
        A non-zero desired velocity drives the error rate proportionally.
        """
        J_m = ur5.jacobian(np.zeros(6))
        theta_dot_p = np.zeros(6)
        delta_V = np.zeros(6)
        V_ed = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        e_dot = ctrl_const.pose_error_dynamics(J_m, theta_dot_p, delta_V, V_ed)
        np.testing.assert_allclose(e_dot, -V_ed, atol=1e-12)


# ---------------------------------------------------------------------------
# Control law
# ---------------------------------------------------------------------------

class TestComputeJointVelocity:
    def test_output_shape(self, ctrl_const, ur5):
        theta = np.zeros(6)
        epsilon_e = np.zeros(6)
        V_ed = np.zeros(6)
        theta_dot_p, J_m = ctrl_const.compute_joint_velocity(
            theta, epsilon_e, V_ed
        )
        assert theta_dot_p.shape == (6,)
        assert J_m.shape == (6, 6)

    def test_zero_error_zero_velocity_gives_zero_output(self, ctrl_const, ur5):
        """When εₑ=0 and V_ed=0, the controller should output zero velocity."""
        theta = np.array([0.1, -0.5, 0.3, -0.2, 0.4, -0.1])
        theta_dot_p, _ = ctrl_const.compute_joint_velocity(
            theta, np.zeros(6), np.zeros(6)
        )
        np.testing.assert_allclose(theta_dot_p, np.zeros(6), atol=1e-8)

    def test_nonzero_error_gives_nonzero_velocity(self, ctrl_const, ur5):
        theta = np.array([0.1, -0.5, 0.3, -0.2, 0.4, -0.1])
        epsilon_e = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        theta_dot_p, _ = ctrl_const.compute_joint_velocity(
            theta, epsilon_e, np.zeros(6)
        )
        assert np.linalg.norm(theta_dot_p) > 1e-6

    def test_default_delta_V_is_zero(self, ctrl_const, ur5):
        """Calling without delta_V should give the same result as delta_V=zeros."""
        theta = np.array([0.1, -0.5, 0.3, -0.2, 0.4, -0.1])
        epsilon_e = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        V_ed = np.zeros(6)

        tdp_no_dv, _ = ctrl_const.compute_joint_velocity(theta, epsilon_e, V_ed)
        tdp_zero_dv, _ = ctrl_const.compute_joint_velocity(
            theta, epsilon_e, V_ed, delta_V=np.zeros(6)
        )
        np.testing.assert_allclose(tdp_no_dv, tdp_zero_dv, atol=1e-12)

    def test_explicit_time_changes_output(self, ctrl_explicit, ur5):
        """Near T_f the time-varying gain produces a larger correction."""
        theta = np.array([0.1, -0.5, 0.3, -0.2, 0.4, -0.1])
        epsilon_e = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
        V_ed = np.zeros(6)

        tdp_early, _ = ctrl_explicit.compute_joint_velocity(
            theta, epsilon_e, V_ed, t=0.0
        )
        tdp_late, _ = ctrl_explicit.compute_joint_velocity(
            theta, epsilon_e, V_ed, t=4.9
        )
        # Near T_f the gain is large → larger joint velocity correction
        assert np.linalg.norm(tdp_late) > np.linalg.norm(tdp_early)


# ---------------------------------------------------------------------------
# Lyapunov diagnostics
# ---------------------------------------------------------------------------

class TestLyapunovDiagnostics:
    def test_lyapunov_value_zero_for_zero_error(self, ctrl_const):
        V = ctrl_const.lyapunov_value(np.zeros(6))
        assert V == pytest.approx(0.0)

    def test_lyapunov_value_positive(self, ctrl_const):
        V = ctrl_const.lyapunov_value(np.ones(6))
        assert V > 0.0

    def test_lyapunov_value_formula(self, ctrl_const):
        e = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        V = ctrl_const.lyapunov_value(e)
        assert V == pytest.approx(0.5 * (1.0 + 4.0))

    def test_lyapunov_derivative_negative_under_control(self, ctrl_const, ur5):
        """
        Under the Lyapunov control law with zero ΔV, 𝒱̇ should be ≤ 0.
        """
        theta = np.array([0.1, -0.5, 0.3, -0.2, 0.4, -0.1])
        epsilon_e = np.array([0.1, 0.05, 0.0, 0.0, 0.0, 0.0])
        V_ed = np.zeros(6)

        theta_dot_p, J_m = ctrl_const.compute_joint_velocity(
            theta, epsilon_e, V_ed
        )
        V_dot = ctrl_const.lyapunov_derivative(
            epsilon_e, theta_dot_p, J_m, np.zeros(6), V_ed
        )
        assert V_dot <= 1e-10  # numerically non-positive


# ---------------------------------------------------------------------------
# Closed-loop convergence test
# ---------------------------------------------------------------------------

class TestClosedLoopConvergence:
    """
    End-to-end convergence test: simulate the controller and check that
    ‖εₑ(t)‖ decreases monotonically and reaches a small value.
    """

    def _simulate(self, ctrl, theta_init, theta_desired, T_total=6.0, dt=0.05):
        robot = ctrl.robot
        T_des = robot.forward_kinematics(theta_desired)
        V_ed = np.zeros(6)
        theta = theta_init.copy()
        epsilon_norms = []

        n_steps = int(np.ceil(T_total / dt))
        for k in range(n_steps):
            t = k * dt
            T_cur = robot.forward_kinematics(theta)
            epsilon_e = robot.pose_error(T_cur, T_des)
            epsilon_norms.append(np.linalg.norm(epsilon_e))
            theta_dot_p, _ = ctrl.compute_joint_velocity(
                theta, epsilon_e, V_ed, t=t
            )
            theta = theta + dt * theta_dot_p

        return np.array(epsilon_norms)

    def test_constant_gain_convergence(self, ur5):
        ctrl = CartesianSpaceController(ur5, K=2.0, T_f=None, damping=1e-3)
        theta_init = np.array([0.3, -1.2, 0.8, -1.0, 0.5, 0.2])
        theta_des = np.zeros(6)
        norms = self._simulate(ctrl, theta_init, theta_des, T_total=6.0, dt=0.05)
        # Error should decrease significantly
        assert norms[-1] < norms[0] * 0.01

    def test_explicit_time_convergence(self, ur5):
        T_f = 5.0
        ctrl = CartesianSpaceController(ur5, K=1.0, T_f=T_f, damping=1e-3)
        theta_init = np.array([0.3, -1.2, 0.8, -1.0, 0.5, 0.2])
        theta_des = np.zeros(6)
        # Simulate up to T_f
        norms = self._simulate(ctrl, theta_init, theta_des, T_total=T_f, dt=0.02)
        # Error at T_f should be very small
        assert norms[-1] < 1e-2

    def test_lyapunov_function_non_increasing(self, ur5):
        """𝒱(t) should be non-increasing throughout the simulation."""
        ctrl = CartesianSpaceController(ur5, K=2.0, T_f=None, damping=1e-3)
        theta_init = np.array([0.3, -1.2, 0.8, -1.0, 0.5, 0.2])
        theta_des = np.zeros(6)
        T_des = ur5.forward_kinematics(theta_des)
        V_ed = np.zeros(6)
        theta = theta_init.copy()

        lyap_vals = []
        dt = 0.05
        for k in range(80):
            t = k * dt
            T_cur = ur5.forward_kinematics(theta)
            epsilon_e = ur5.pose_error(T_cur, T_des)
            lyap_vals.append(ctrl.lyapunov_value(epsilon_e))
            theta_dot_p, _ = ctrl.compute_joint_velocity(theta, epsilon_e, V_ed, t=t)
            theta = theta + dt * theta_dot_p

        lyap_vals = np.array(lyap_vals)
        # 𝒱(t) should never increase by more than a small numerical tolerance
        diffs = np.diff(lyap_vals)
        assert np.all(diffs <= 1e-6), (
            f"Lyapunov function increased at steps: {np.where(diffs > 1e-6)[0]}"
        )
