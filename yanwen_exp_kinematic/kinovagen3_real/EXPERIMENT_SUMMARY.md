# Kinova Gen3 Real-Machine Experiment Summary

## Experiment Source

Date/run folder:

```text
trajectory_results/run-20260603-142141
```

Original location:

```text
/home/kinova/ssd1/YanWen/Kinova-kortex2_Gen3_G3L/api_python/examples/Explicit_time_kinematic/trajectory_results/run-20260603-142141
```

Packaged location:

```text
results/run-20260603-142141
```

Additional real-machine image:

```text
results/experiment_photo_20260603.jpg
```

## Implemented Method

The real-machine code implements the analytical pose-feedback explicit-time kinematic controller:

```text
e_p = p_e - p_ed
R_err = R_e R_ed^T
phi = log(R_err)
phi_dot = J_l^{-1}(phi) (omega_e - R_err omega_ed)
J_a = diag(I, J_l^{-1}(phi)) J_m
theta_dot = J_a^+ V_edp
```

Important conventions:

- pose attitude uses fixed `ZYX` Euler angles, stored as `[Z, Y, X]`;
- tool offset is `z_tool = -0.16746 m`;
- the current source is synchronized with the corrected analytical attitude/Jacobian relation used in the Isaac demo.

## Packaged Result Plots

- `results/run-20260603-142141/ee_6d_error_tracking.png`: end-effector 6D tracking/error plot from the real robot run.
- `results/run-20260603-142141/joint_velocity_curves.png`: joint velocity command curves from the real robot run.
- `results/experiment_photo_20260603.jpg`: real-machine experiment screenshot/photo.

## What Is Included

- Real-robot source code copied from `Explicit_time_kinematic`.
- The 2026-06-03 result plots.
- The provided real-machine image.
- Safety and usage notes in `README.md`.

## Current Limitation

No raw numeric CSV/NPZ log was found in the source run folder. The packaged real-machine evidence is therefore based on the saved plots and image, while the Isaac side includes both plots and `data/offline_planner_trace.npz`.

## Recommended Next Logging Improvement

For future real-robot runs, save a numeric log such as:

```text
time, theta_actual[7], theta_dot_cmd[7], P_e[3], Phi_e[3], P_target[3], Phi_target[3], position_error, attitude_error
```

This will allow direct quantitative comparison between Isaac and the Kinova Gen3 real-machine run.
