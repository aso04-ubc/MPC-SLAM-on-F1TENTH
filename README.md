This is the branch that can successfully run in sim. Replace levine map with `./levine.png` to have best performace.

To run in sim, use 
```ros2 launch milestones race_line_stack.launch.py sim:=true odom_topic:=/ego_racecar/odom map_window_size:=1000 <- (change it based on the setting of the sim map)```

It really requires a lot of computation power, the larger the map is more power is needed. I have tested using 16 cores of cpu and 10 gige of memory, it should be able to run under 1000 px map.
