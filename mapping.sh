#! /bin/sh 



echo "Opening world"
tmux new -s map -d "ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py"
sleep 3
echo "mapping launching"
tmux new -s pcl -d "ros2 run gmapper gmap"
sleep 3
echo "Visualizer open"
tmux new -s rviz -d "ros2 launch turtlebot3_bringup rviz2.launch.py"

