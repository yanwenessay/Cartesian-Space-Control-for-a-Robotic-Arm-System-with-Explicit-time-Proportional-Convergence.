# Realtime Pose Web Monitor

启动实时末端位姿 Web 监控：

```bash
python3 realtime_pose_web.py --ip <ROBOT_IP> -u <USERNAME> -p <PASSWORD> --web-host <WEB_BIND_HOST> --web-port 8088 --poll-hz 20
```

然后在浏览器打开：

```text
http://<WEB_BIND_HOST>:8088
```

JSON API：

```text
http://<WEB_BIND_HOST>:8088/api/pose
```

不要把真实机械臂 IP、登录信息或现场主机地址提交到公开仓库。
