# Kinova Gen3 Explicit-Time Kinematic Planning

这是一个面向 Kinova Gen3 7 自由度机械臂的显式时间运动学规划与在线控制示例包。代码包含末端位姿规划、逆运动学、在线关节速度控制、waypoint 速度轨迹执行，以及一个可选的 Web 位姿监控页面。

## 文件结构

```text
explicit_time_kinematic_open_source/
  planning_main.py                    # 显式时间规划入口函数
  inverse_kinematics.py               # 逆运动学
  Kinematic_fcn.py                    # 正运动学/Jacobian 相关函数
  Yan_vel_control.py                  # 末端速度控制
  pose_end_planning.py                # 实机末端位姿闭环控制
  online_joint_speed_control.py       # 单目标在线关节速度控制
  waypoint_speed_trajectory.py        # 多 waypoint 速度轨迹
  kinova_arm_verification.py          # 机械臂验证流程
  test_main.py                        # 离线测试/绘图
  utilities.py                        # 已清理默认 IP 的 Kortex 连接工具
  Web_view/realtime_pose_web.py       # 实时位姿 Web 监控
```

历史运行结果、日志、缓存和私有网络信息没有放入这个开源目录。

## 安全提醒

这些脚本会向真实机械臂发送关节速度或关节角命令。运行前请确认：

- 机械臂周围没有人员和障碍物。
- 急停、示教器和安全限位可用。
- 机械臂处于无 fault 状态。
- 已按自己的末端工具长度、目标点、速度上限重新检查源码中的参数。
- 第一次运行建议降低 `MAX_JOINT_SPEED_DEG_S`、`SPEED_SCALE`、`MAX_TOTAL_SECONDS`，并随时准备急停。

## 安装

建议把本目录放在 Kinova Kortex Python API 的 `api_python/examples/` 目录下，例如：

```text
api_python/examples/explicit_time_kinematic_open_source/
```

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

如果你已经按 Kinova 官方方式安装了 `kortex_api`，可以只安装：

```bash
python3 -m pip install numpy scipy matplotlib
```

## 运行示例

本开源包不会保存真实机械臂 IP、用户名或密码。所有实机脚本都需要用户运行时自行填写：

```bash
python3 pose_end_planning.py --ip <ROBOT_IP> -u <USERNAME> -p <PASSWORD>
```

多 waypoint 速度轨迹：

```bash
python3 waypoint_speed_trajectory.py --ip <ROBOT_IP> -u <USERNAME> -p <PASSWORD>
```

在线关节速度控制：

```bash
python3 online_joint_speed_control.py --ip <ROBOT_IP> -u <USERNAME> -p <PASSWORD>
```

机械臂验证流程：

```bash
python3 kinova_arm_verification.py --ip <ROBOT_IP> -u <USERNAME> -p <PASSWORD>
```

离线测试不需要连接机械臂：

```bash
python3 test_main.py
```

## 常用参数位置

当前版本保留了原始脚本的参数组织方式，便于对照论文/实验代码：

- `constant.py`: 末端工具长度 `z_tool`。
- `pose_end_planning.py`: `P_target`、`Phi_target_deg`、`dt`、`max_joint_velocity`、收敛阈值。
- `online_joint_speed_control.py`: `P_TARGET`、`PHI_TARGET`、`MAX_JOINT_SPEED_DEG_S`、`SPEED_SCALE`、`MAX_TOTAL_SECONDS`。
- `waypoint_speed_trajectory.py`: `WAYPOINTS`、`SEGMENT_DURATION_S`、`MAX_JOINT_SPEED_DEG_S`、`SPEED_SCALE`。
- `Web_view/realtime_pose_web.py`: `Z_TOOL`、`DEFAULT_POLL_HZ`、`DEFAULT_WEB_PORT`。

## Web 位姿监控

```bash
cd Web_view
python3 realtime_pose_web.py --ip <ROBOT_IP> -u <USERNAME> -p <PASSWORD> --web-host <WEB_BIND_HOST> --web-port 8088 --poll-hz 20
```

浏览器打开：

```text
http://<WEB_BIND_HOST>:8088
```

API：

```text
http://<WEB_BIND_HOST>:8088/api/pose
```

## 开源隐私说明

本目录不包含真实机械臂 IP、登录用户名、密码、现场主机地址、运行日志或历史结果图。公开仓库中请继续避免提交：

- `trajectory_results/`
- `*.log`
- 本地配置文件
- 带有现场网络地址的截图或 README 片段
