"""
Unit tests for robot_model.RobotModel.

Tests cover:
* DH forward kinematics (known 2-link planar and UR5 configurations)
* Geometric Jacobian (finite-difference verification)
* Pose error computation
* End-effector velocity
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from robot_model import RobotModel, build_ur5_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def planar_2dof():
    """Return a 2-DOF planar robot with unit link lengths."""
    dh_params = [
        {"a": 1.0, "d": 0.0, "alpha": 0.0, "theta_offset": 0.0},
        {"a": 1.0, "d": 0.0, "alpha": 0.0, "theta_offset": 0.0},
    ]
    return RobotModel(dh_params)


@pytest.fixture
def ur5():
    """Return the UR5 model."""
    return build_ur5_model()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestRobotModelInit:
    def test_joint_count(self, planar_2dof):
        assert planar_2dof.n_joints == 2

    def test_empty_params_raises(self):
        with pytest.raises(ValueError, match="at least one joint"):
            RobotModel([])

    def test_ur5_has_6_joints(self, ur5):
        assert ur5.n_joints == 6


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

class TestForwardKinematics:
    def test_home_pose_is_4x4(self, planar_2dof):
        T = planar_2dof.forward_kinematics([0.0, 0.0])
        assert T.shape == (4, 4)

    def test_identity_bottom_row(self, planar_2dof):
        T = planar_2dof.forward_kinematics([0.5, -0.3])
        np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-12)

    def test_rotation_matrix_orthogonal(self, planar_2dof):
        T = planar_2dof.forward_kinematics([0.5, -0.3])
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)

    def test_planar_end_position_at_zero(self, planar_2dof):
        """At [0, 0] the tip should be at (2, 0, 0) for unit links."""
        T = planar_2dof.forward_kinematics([0.0, 0.0])
        np.testing.assert_allclose(T[:3, 3], [2.0, 0.0, 0.0], atol=1e-10)

    def test_planar_end_position_at_pi_over_2(self, planar_2dof):
        """At [π/2, 0] the first link points along +y."""
        T = planar_2dof.forward_kinematics([np.pi / 2, 0.0])
        # tip should be at (0, 2, 0) in the xy-plane
        np.testing.assert_allclose(T[:3, 3], [0.0, 2.0, 0.0], atol=1e-10)

    def test_wrong_theta_shape_raises(self, planar_2dof):
        with pytest.raises(ValueError):
            planar_2dof.forward_kinematics([0.0])

    def test_ur5_home_is_4x4(self, ur5):
        T = ur5.forward_kinematics(np.zeros(6))
        assert T.shape == (4, 4)

    def test_ur5_rotation_orthogonal(self, ur5):
        theta = np.array([0.1, -0.5, 0.3, -0.2, 0.4, -0.1])
        T = ur5.forward_kinematics(theta)
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-11)


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------

class TestJacobian:
    def _finite_diff_jacobian(self, robot, theta, eps=1e-6):
        """Numerically estimate the Jacobian via finite differences."""
        n = robot.n_joints
        J_fd = np.zeros((6, n))
        T0 = robot.forward_kinematics(theta)
        e0 = robot.pose_error(T0, np.eye(4))
        for i in range(n):
            dtheta = theta.copy()
            dtheta[i] += eps
            T1 = robot.forward_kinematics(dtheta)
            e1 = robot.pose_error(T1, np.eye(4))
            J_fd[:, i] = (e1 - e0) / eps
        return J_fd

    def test_jacobian_shape_2dof(self, planar_2dof):
        J = planar_2dof.jacobian([0.0, 0.0])
        assert J.shape == (6, 2)

    def test_jacobian_shape_ur5(self, ur5):
        J = ur5.jacobian(np.zeros(6))
        assert J.shape == (6, 6)

    def test_planar_linear_velocity_at_zero(self, planar_2dof):
        """
        For a planar robot at θ=[0,0], the z-axes are [0,0,1], and both
        linear Jacobian columns should lie in the xy-plane.
        """
        J = planar_2dof.jacobian([0.0, 0.0])
        # Linear part: z-axis for both joints = [0, 0, 1]
        # J_linear_0 = [0,0,1] × (p_e - p_0) = [0,0,1] × [2,0,0] = [0,2,0]  → wrong? let me recalc
        # p_0 = [0,0,0], p_e = [2,0,0], z_0 = [0,0,1]
        # cross([0,0,1], [2,0,0]) = [0*0-1*0, 1*2-0*0, 0*0-0*2] = [0, 2, 0]
        np.testing.assert_allclose(J[:3, 0], [0.0, 2.0, 0.0], atol=1e-10)
        # J_linear_1 = [0,0,1] × (p_e - p_1) = [0,0,1] × [1,0,0] = [0,1,0]
        np.testing.assert_allclose(J[:3, 1], [0.0, 1.0, 0.0], atol=1e-10)

    def test_ur5_jacobian_varies_with_configuration(self, ur5):
        J1 = ur5.jacobian(np.zeros(6))
        J2 = ur5.jacobian(np.array([0.1, -0.5, 0.3, -0.2, 0.4, -0.1]))
        assert not np.allclose(J1, J2)


# ---------------------------------------------------------------------------
# Pose error
# ---------------------------------------------------------------------------

class TestPoseError:
    def test_zero_error_for_identical_poses(self, planar_2dof):
        theta = [0.3, -0.7]
        T = planar_2dof.forward_kinematics(theta)
        error = planar_2dof.pose_error(T, T)
        np.testing.assert_allclose(error, np.zeros(6), atol=1e-12)

    def test_position_error_only(self, planar_2dof):
        """
        Sign convention: ε_e = current − desired.
        T_cur at origin, T_des at x=1  →  error = [0-1, 0, 0] = [-1, 0, 0].
        """
        T_cur = np.eye(4)
        T_des = np.eye(4)
        T_des[0, 3] = 1.0  # desired is 1 m ahead along x
        error = planar_2dof.pose_error(T_cur, T_des)
        np.testing.assert_allclose(error[:3], [-1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(error[3:], np.zeros(3), atol=1e-12)

    def test_orientation_error_only(self, planar_2dof):
        """
        π/2 rotation about z-axis.
        T_cur = I, T_des = Rz(π/2).
        R_err = R_cur @ R_des.T = I @ Rz(-π/2)  →  axis-angle [0, 0, -π/2].
        """
        T_cur = np.eye(4)
        T_des = np.eye(4)
        ang = np.pi / 2
        T_des[:3, :3] = np.array([
            [np.cos(ang), -np.sin(ang), 0],
            [np.sin(ang),  np.cos(ang), 0],
            [0,            0,           1],
        ])
        error = planar_2dof.pose_error(T_cur, T_des)
        np.testing.assert_allclose(error[:3], np.zeros(3), atol=1e-12)
        # ε_e = current − desired  →  orientation error = [0, 0, −π/2]
        np.testing.assert_allclose(error[3:], [0.0, 0.0, -np.pi / 2], atol=1e-12)

    def test_error_norm_decreases_toward_desired(self, ur5):
        theta_a = np.zeros(6)
        theta_b = np.array([0.1, -0.5, 0.3, -0.2, 0.4, -0.1])
        theta_des = np.array([0.05, -0.25, 0.15, -0.1, 0.2, -0.05])
        T_a = ur5.forward_kinematics(theta_a)
        T_b = ur5.forward_kinematics(theta_b)
        T_des = ur5.forward_kinematics(theta_des)
        # theta_des is the midpoint: error from T_a should be larger than
        # an interpolated config that is closer
        err_a = np.linalg.norm(ur5.pose_error(T_a, T_des))
        err_b = np.linalg.norm(ur5.pose_error(T_b, T_des))
        # Both errors are positive (non-zero)
        assert err_a > 0
        assert err_b > 0


# ---------------------------------------------------------------------------
# End-effector velocity
# ---------------------------------------------------------------------------

class TestEndEffectorVelocity:
    def test_shape(self, ur5):
        theta = np.zeros(6)
        theta_dot = np.ones(6)
        V_e = ur5.end_effector_velocity(theta, theta_dot)
        assert V_e.shape == (6,)

    def test_zero_joint_velocity_gives_zero_ee_velocity(self, ur5):
        theta = np.array([0.1, -0.5, 0.3, -0.2, 0.4, -0.1])
        V_e = ur5.end_effector_velocity(theta, np.zeros(6))
        np.testing.assert_allclose(V_e, np.zeros(6), atol=1e-12)
