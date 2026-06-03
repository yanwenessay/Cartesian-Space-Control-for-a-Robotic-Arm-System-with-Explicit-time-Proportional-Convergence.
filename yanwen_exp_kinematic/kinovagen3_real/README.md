# Kinova Gen3 Real-Robot Explicit-Time Kinematic Experiment

## Source Code

`source_explicit_time_kinematic/` is copied from:

```text
/home/kinova/ssd1/YanWen/Kinova-kortex2_Gen3_G3L/api_python/examples/Explicit_time_kinematic
```

Core files:

- `Yan_vel_control.py`: analytical pose-feedback law using `R_err = R_e R_ed^T` and `J_l^{-1}`.
- `inverse_kinematics.py`: damped least-squares IK; supports analytical transform `Lambda_e`.
- `planning_main.py`: planner entry point, computes `V_edp` and joint velocity command.
- `Kinematic_fcn.py`: Kinova Gen3 DH forward kinematics and ZYX Euler pose output.
- `waypoint_speed_trajectory.py`, `pose_end_planning.py`, `online_joint_speed_control.py`: real-robot control/demo scripts using Kortex API.

## Real-Machine Result: 2026-06-03

Copied from:

```text
/home/kinova/ssd1/YanWen/Kinova-kortex2_Gen3_G3L/api_python/examples/Explicit_time_kinematic/trajectory_results/run-20260603-142141
```

Included result plots:

```text
results/run-20260603-142141/ee_6d_error_tracking.png
results/run-20260603-142141/joint_velocity_curves.png
results/experiment_photo_20260603.jpg
```

Image dimensions:

- `ee_6d_error_tracking.png`: 1600 x 800
- `joint_velocity_curves.png`: 1400 x 900
- `experiment_photo_20260603.jpg`: 1706 x 1279

## Summary

This real-machine experiment verifies the same analytical pose-feedback planning structure used in the Isaac release:

```text
P_e, Phi_e = FK(theta)
R_err = R_e R_ed^T
phi = log(R_err)
V_edp = V_ed,a - K [p_e - p_ed, phi]
theta_dot = J_a^+ V_edp
J_a = diag(I, J_l^{-1}(phi)) J_m
```

The available saved results show:

- end-effector 6D tracking/error behavior in `ee_6d_error_tracking.png`;
- seven-joint velocity commands in `joint_velocity_curves.png`;
- real robot experimental setup/status in `experiment_photo_20260603.jpg`.

No raw numeric CSV/NPZ log was found in `run-20260603-142141`; this folder currently preserves the source code and plotted real-machine results.

## Safety Notes

- Confirm Kortex connection IP, username, password, and servoing mode before execution.
- Check `z_tool = -0.16746` against the mounted end-effector/tool.
- Keep joint speed limits enabled for real-robot tests unless there is a specific supervised reason to disable them.
- Verify the `ZYX` Euler order and target pose convention before comparing with Isaac plots.
