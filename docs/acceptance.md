# MPC Acceptance

The implementation is considered acceptable when:

1. The ROS2 node builds and starts as `mpc_controller_node`.
2. With valid odometry and a valid path CSV, the node forms an LTV tracking problem and calls the QP solver every control tick.
3. The node publishes raw Ackermann commands on `/mpc/drive`.
4. The node publishes priority-tagged commands on `/drive_control` when `dev_b7_interfaces` is available.
5. Missing odometry, stale odometry, invalid reference data, or missing solver solutions result in a safe stop.
6. Pure Python helper logic for path loading and tracking-problem assembly passes the repository tests.
