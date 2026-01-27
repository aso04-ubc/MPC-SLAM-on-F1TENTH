source /opt/ros/foxy/setup.bash

if [ -f "./install/setup.bash" ]; then
    source ./install/setup.bash
    echo "success"
else
    echo "./install/setup.bash not found. cd into src folder"
fi