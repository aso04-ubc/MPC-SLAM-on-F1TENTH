# MPC #

Describes what we need to do

## To-do ##

**Minimum-Time Optimization:** Time based optimization, requires a track map
* **Dynamic Model:** Account for vehicle sliding
    + **Tire slip angle:** Difference between car travel heading and steering angle
    + **Pacejka Formulae:** Calculate maximum friction before loss of grip
    + **Weight Transfer:** Account for vehicle acceleration changing the car's center of mass and grip profile
* **Three-pronged approach:** Three separate nodes to reduce latency, each node has its own package to facilitate testing (building one node instead of three), and to manage dependency differences between different nodes
    + **Perception and State Estimation:** Receives IMU, odometry, LiDAR, camera input, high refresh rate
    + **Path Planner:** Keep track of track limits and reference points, static, avoid constant reloads of map
    + **Control:** Receives data from other two nodes, executes optimizer code
* **Build the map (required):** Use a gap follow algorithm to record a path and smooth LiDAR points into continuous boundaries


### Easier Alternatives ###

**Reference Path Tracking:** Optimize for the most effective path, considering grip levels  
**Dynamic Window Approach:** Optimize cost function (time) based on obstacle position and velocity