import constant as cont
import inverse_kinematics
import Kinematic_fcn
import Yan_vel_control
import trapez_vel_plan
import constant as cont

def planning(P_ed, Phi_ed, theta): #return theta_dot_planning
    
    V_ed=[0,0,0,0,0,0]

    V_edp, p_error, delta_error = Yan_vel_control.vel_control(V_ed, theta, P_ed, Phi_ed) #调用控制

    theta_dot_planning, Sigma = inverse_kinematics.inverse_kinematics(theta, V_edp, cont.z_tool) #调用逆运动学程序

    return theta_dot_planning