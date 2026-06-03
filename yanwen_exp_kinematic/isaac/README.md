# Analytical Pose-Feedback Planner Usage

## 1. Offline Kinematic Demo

Run the standalone explicit-time planner demo without Isaac Sim:

```bash
cd /home/kinova/ssd1/YanWen/yanwen_exp_kinematic/isaac
python3 examples/offline_convergence_demo.py
```

Generated data:

```text
data/offline_planner_trace.npz
```

Generated figures:

```text
figures/pose_error_convergence.png
figures/pose_error_components.png
figures/joint_velocity_commands.png
figures/joint_angle_history.png
figures/analytical_jacobian_singular_values.png
figures/end_effector_path_3d.png
```

The demo runs for a fixed 10 seconds and does not stop early by an error threshold. It uses two target poses: target 1 from 0-5 s and target 2 from 5-10 s. Joint angle plots are shown in the normalized [-180 deg, 180 deg] range, with wrap discontinuities broken instead of connected.

## 2. Isaac Kinova Gen3 3D Video

The Isaac video is driven by the same `data/offline_planner_trace.npz` trajectory, not by random or decorative joint commands. Recording starts at -2 s with a static pre-roll, then planner time 0-5 s moves to target 1 and planner time 5-10 s moves to target 2.

Existing output:

```text
videos/kinova_gen3_isaac_motion.mp4
figures/kinova_gen3_isaac_video_preview.jpg
```

Regenerate it on this machine with:

```bash
source /home/kinova/ssd1/YanWen/isaac_lab/use_isaaclab_lhm.sh
cd /home/kinova/ssd1/YanWen/isaac_lab/_runtime/IsaacLab_lhm
./isaaclab.sh -p /home/kinova/ssd1/YanWen/yanwen_exp_kinematic/isaac/source_isaac_sim/record_kinova_gen3_video.py \
  --headless --enable_cameras --pre-roll 2 --fps 24 --width 960 --height 540 --crf 31
```

Optional camera adjustment:

```bash
./isaaclab.sh -p /home/kinova/ssd1/YanWen/yanwen_exp_kinematic/isaac/source_isaac_sim/record_kinova_gen3_video.py \
  --headless --enable_cameras \
  --camera-eye 2.25 -2.55 1.70 \
  --camera-target 0.0 0.0 0.48
```

The default H.264 compression is intended to keep the video below 5 MB for GitHub.

## 3. Optional Lightweight Preview

If Isaac Lab is unavailable, generate a simple kinematic preview:

```bash
cd /home/kinova/ssd1/YanWen/yanwen_exp_kinematic/isaac
python3 examples/create_motion_preview_video.py
```

Output:

```text
videos/kinova_motion_preview.mp4
```

This fallback is only a lightweight visualization and is not an Isaac render.

## Notes

- Euler angles use fixed ZYX order and are stored as `[Z, Y, X]`.
- The pose error convention is `e = p_e - p_ed` and `R_err = R_e R_ed^T`.
- The planner uses `J_a = diag(I, J_l^{-1}(phi)) J_m`.
- The release demos do not clip joint velocity commands. Add safety limits before real robot experiments.
- The offline model uses the DH parameters copied from `Explicit_time_kinematic/Kinematic_fcn.py`.
- The Isaac video uses Isaac Lab's built-in `KINOVA_GEN3_N7_CFG` asset at `Robots/Kinova/Gen3/gen3n7_instanceable.usd`.
- For the video, Isaac joints are replayed kinematically with `write_joint_state_to_sim` to avoid PD-controller shaking.
- Before publishing quantitative Isaac Sim results, verify USD joint order, zero positions, joint axes, and tool frame against the DH model.
