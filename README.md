# MPC #

Describes what we need to do

## Options ##

1. **Reference Path Tracking:** Optimize for the most effective path, considering grip levels
2. **Dynamic Window Approach:** Optimize cost function (time) based on obstacle position and velocity
3. **Minimum-Time Optimization:** Time based optimization (most difficult)
    * Use Dynamic Model to account for vehicle sliding
        + **Tire slip angle:** Difference between car travel heading and steering angle
        + **Pacejka Formulae:** Calculate maximum friction before loss of grip
        + **Weight Transfer:** Account for vehicle acceleration changing the car's center of mass and grip profile