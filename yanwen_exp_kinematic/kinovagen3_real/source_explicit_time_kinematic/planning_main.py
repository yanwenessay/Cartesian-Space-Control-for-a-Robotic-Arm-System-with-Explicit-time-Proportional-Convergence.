import constant as cont
import inverse_kinematics
import Yan_vel_control


def planning(P_ed, Phi_ed, theta):
    """Explicit-time pose-feedback planner using analytical Jacobian Ja."""
    V_ed = [0, 0, 0, 0, 0, 0]
    V_edp, p_error, delta_error, Lambda_e = Yan_vel_control.vel_control(
        V_ed, theta, P_ed, Phi_ed
    )
    theta_dot_planning, Sigma = inverse_kinematics.inverse_kinematics(
        theta, V_edp, cont.z_tool, analytical_transform=Lambda_e
    )
    return theta_dot_planning
