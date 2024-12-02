# ros2_gmapping

**Gmapping ROS 2 Implementation**

## Overview
This repository contains a **ROS 2** implementation of **Gmapping**, an approach for simultaneous localization and mapping (**SLAM**). The key feature of this implementation is the use of **Rao-Blackwellized Particle Filters (RBPF)** to perform mapping while simultaneously tracking the pose of the robot. The framework leverages the **GTSAM** library to optimize the robot's pose and improve the accuracy of mapping through factor graph optimization.

## Rao-Blackwellized Particle Filters (RBPF)

The core of this SLAM approach lies in the use of **Rao-Blackwellized Particle Filters**. In a traditional particle filter, every particle represents a possible state of the system, including both the robot's pose and the map it builds. In RBPF, the problem is divided to reduce computational complexity:

- **Pose Estimation**: Particle filters are used to track the robot's pose.
- **Map Estimation**: Given the estimated poses, the map is generated.

This division enables a more efficient representation of the uncertainty inherent in SLAM. The **RBPF** approach improves over the standard particle filter by reducing the variance of estimates and focusing computational resources on tracking the robot's pose, making it especially useful for complex environments.

### Mathematical Representation

In this implementation, the **Rao-Blackwellized Particle Filter** is used to represent the posterior distribution of the robot pose and map as:

P(x<sub>t</sub>, m | z<sub>1:t</sub>, u<sub>1:t-1</sub>) = P(x<sub>t</sub> | z<sub>1:t</sub>, u<sub>1:t-1</sub>) * P(m | x<sub>1:t</sub>, z<sub>1:t</sub>)

Where:
- **x<sub>t</sub>** represents the robot pose at time **t**.
- **m** represents the map.
- **z<sub>1:t</sub>** are all sensor measurements from the beginning until time **t**.
- **u<sub>1:t-1</sub>** are all odometry readings from the beginning until time **t-1**.

The particle filter estimates the robot's pose (**x<sub>t</sub>**), while a separate map estimation step is performed for each particle to incrementally build the map (**m**). In this implementation, particles are visualized as small arrows or dots around the robot's estimated position in **RViz** to illustrate the distribution of possible robot poses.

### Implementation Details

This project includes:
- **Motion Model**: Uses odometry information to predict the robot's next pose, incorporating noise for uncertainty.
- **Measurement Model**: Uses laser scan data to adjust weights of particles based on how well the predicted map matches the current scan. Scan matching is performed using the **ICP (Iterative Closest Point)** algorithm.
- **Occupancy Grid Map**: The occupancy grid map is continuously updated and published to **/map** for visualization in **RViz**. Each cell's probability is represented in log-odds to facilitate efficient updates.

## GTSAM Integration

**GTSAM (Georgia Tech Smoothing and Mapping)** is a powerful library for factor graph-based optimization. It has been integrated into both the **motion model** and **measurement model** to improve accuracy and consistency.

### GTSAM in Motion Model

The **motion model** in this implementation predicts the robot's movement based on odometry, but the odometry data inherently contains noise and drift. By using **GTSAM**, a factor graph is constructed to add constraints between poses over time. These constraints help optimize the robot's trajectory through **pose graph optimization**.

- **Pose Graph**: Each time step is represented as a node in the graph, and odometry is used to create a connection (edge) between consecutive poses. This graph is periodically optimized using **Levenberg-Marquardt** or other optimization techniques to minimize error across the trajectory.
- **Impact on Particle Filter**: GTSAM optimization reduces uncertainty in pose estimates, thus making the subsequent particle filter updates more robust. The particles receive more accurate initial guesses for their poses, allowing the filter to converge faster and build a more reliable map.

### GTSAM in Measurement Model

The **measurement model** utilizes **GTSAM** to refine the pose estimates using **laser scan data**. After each scan, an optimization step is performed to align the robot's predicted pose with the observed environment:

- **ICP-based Scan Matching**: The **Iterative Closest Point** (ICP) algorithm is used to match current scans to previous scans or a known map. The **scan matching** process generates a transformation that is incorporated into the factor graph as a new constraint, improving the consistency of the map and pose estimates.
- **Pose Update**: By integrating these transformations into the graph, **GTSAM** helps minimize the overall alignment error, leading to more accurate localization and more consistent mapping. This improves the performance of the **Rao-Blackwellized Particle Filter**, as particles are resampled around a more reliable trajectory estimate.

## Mapping Process

### Predict Step (Motion Model)
The **motion model** uses odometry data to propagate the particles forward, adding Gaussian noise to account for the uncertainty in the robot's movement. **GTSAM** is used here to maintain an optimized graph of past poses, allowing corrections to the predicted path and reducing drift.

### Update Step (Measurement Model)
The **measurement model** adjusts the weight of each particle based on laser scan data. **ICP** is used to match the predicted scan against the observed scan, and **GTSAM** refines the estimated transformations, resulting in better alignment between the predicted and actual environment.

### Resampling
The **resampling** step is triggered when the **effective number of particles** falls below a threshold. During this step, particles with higher weights are duplicated, while those with lower weights are removed. The result is a new set of particles distributed according to the updated belief about the robot's pose.

## Visualization in RViz
- **Particles Visualization**: The particles are visualized as arrows to show the distribution of possible robot poses. This helps demonstrate the convergence of the particle filter over time.
- **Occupancy Grid Map**: The occupancy grid is published as a **/map** topic and can be visualized in **RViz**. It provides a representation of the environment, with cells marked as occupied, free, or unknown.

## Parameters Used
The following parameters are used in this ROS 2 implementation:

- **Particles Count (`particles_count`)**: The number of particles used in the filter.
- **Linear Noise (`linear_noise`)**: The standard deviation of the noise applied to the linear motion of the robot.
- **Angular Noise (`angular_noise`)**: The standard deviation of the noise applied to the rotational motion of the robot.
- **Occupancy Grid Parameters**: The size of the grid, resolution, and log-odds parameters used to update occupancy probabilities.

## How to Use TurtleBot3 Simulation in Gazebo

To visualize the mapping process using this **ROS 2** implementation, you can simulate **TurtleBot3** in **Gazebo**. Below is a step-by-step guide to set up and use the simulation environment:

### Prerequisites
Ensure that you have installed the following:
- **ROS 2** (Humble/Foxy, or other supported versions).
- **Gazebo** for simulation.
- **TurtleBot3** packages (`turtlebot3`, `turtlebot3_simulations`).

### Step 1: Set Up Environment Variables
To begin, export the environment variables for **TurtleBot3**. In your terminal, run:
```sh
export TURTLEBOT3_MODEL=burger
```
This sets the **TurtleBot3** model to **Burger**. You can change it to **waffle** or **waffle_pi** if desired.

### Step 2: Launch Gazebo Simulation
To launch the **TurtleBot3** simulation in **Gazebo**, run:
```sh
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```
This command will launch the **Gazebo** environment with **TurtleBot3** placed in a predefined world.

### Step 3: Launch SLAM Node
In a new terminal, launch the **Gmapping** SLAM node from this repository:
```sh
ros2 run ros2_gmapping rbpf_particle_filter
```
Ensure that the **TurtleBot3** and **Gmapping** nodes are communicating over the same ROS 2 network.

### Step 4: Visualize in RViz
To visualize the robot and mapping process, launch **RViz**:
```sh
ros2 launch turtlebot3_bringup rviz2.launch.py
```
In **RViz**, make sure to add the **/map** topic to visualize the occupancy grid and the **/particles** topic to see the particles. You should see **TurtleBot3** navigating and creating a map of the environment in real-time, with particles being displayed to illustrate the localization uncertainty.

### Step 5: Control the Robot
You can control the robot using **teleop** or an autonomous navigation script:
```sh
ros2 run turtlebot3_teleop teleop_keyboard
```
Use the keyboard commands to drive the robot around and watch as the **RBPF** builds a map and tracks the robot's pose.

## Conclusion
This **ROS 2** implementation of **Gmapping** with **Rao-Blackwellized Particle Filters** provides an efficient approach to **SLAM** by separating the pose and map estimation tasks. The integration of **GTSAM** significantly improves the accuracy of both localization and mapping by leveraging factor graph optimization to correct pose estimates, thereby enhancing the performance of the particle filter. The visualization of particles and the occupancy grid in **RViz** provides valuable insight into how the robot builds and understands its environment.

Feel free to explore, modify, and extend this implementation for your SLAM projects!
