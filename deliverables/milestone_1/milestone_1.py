import subprocess
import time

ros_env = "source /opt/ros/foxy/setup.bash"
ws_env = "source ~/sim_ws/install/setup.bash"
env_cmd = f"{ros_env} && {ws_env}"

cmd_drive = f"{env_cmd} && ros2 run safety safety_node -t 0.3"
cmd_wall_follow = f"{env_cmd} && ros2 run wall_follow wall_follow_node"

print("Starting Drive Control node...")
p1 = subprocess.Popen(cmd_drive, shell=True, executable="/bin/bash")

print("Starting Wall Follow node...")
p2 = subprocess.Popen(cmd_wall_follow, shell=True, executable="/bin/bash")

try:
    print("Both nodes are running. Press Ctrl+C to stop.")
    p1.wait()
    p2.wait()
except KeyboardInterrupt:
    print("\nShutting down nodes...")
    p1.terminate()
    p2.terminate()
