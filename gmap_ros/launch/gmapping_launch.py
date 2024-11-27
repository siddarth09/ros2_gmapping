import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Set up environment variables for TurtleBot3 model
    TURTLEBOT3_MODEL = os.environ.get('TURTLEBOT3_MODEL', 'waffle')  # burger as default, can be waffle or waffle_pi
    
    # Get TurtleBot3 package directory
    turtlebot3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    turtlebot3_bringup_dir=get_package_share_directory('turtlebot3_bringup')
    turtlebot3_launch_file = os.path.join(turtlebot3_gazebo_dir, 'launch', 'turtlebot3_world.launch.py')
    
    # Get RViz config file directory (change this path to point to your specific RViz config if needed)
    rviz_config_dir = os.path.join(get_package_share_directory('turtlebot3_navigation2'), 'rviz', 'nav2_default_view.rviz')
    
    # Node to launch the particle filter script
    particle_filter_node = Node(
        package='gmap_ros',
        executable='particle_filter.py',
        name='particle_filter',
        output='screen',
        parameters=[
            {'particles': 3000}  # Modify according to your needs
        ]
    )

    # Launch the TurtleBot3 Gazebo simulation in the turtlebot3_world environment
    turtlebot3_gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(turtlebot3_launch_file),
        launch_arguments={
            'use_sim_time': 'true'  
        }.items(),
    )

    # Launch RViz to visualize the robot, map, and particles
    turtlebot3_rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(turtlebot3_bringup_dir, 'launch', 'rviz2.launch.py')),
        launch_arguments={
            'use_sim_time': 'true' 
            }.items(),
    )

    return LaunchDescription([
        turtlebot3_gazebo_launch,
        particle_filter_node,
        turtlebot3_rviz_launch
    ])
