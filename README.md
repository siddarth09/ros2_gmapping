# ROS2 GMAPPING

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

###TODO: 
## GTSAM Integration

**GTSAM (Georgia Tech Smoothing and Mapping)** is a powerful library for factor graph-based optimization. It has been integrated into both the **motion model** and **measurement model** to improve accuracy and consistency.

### GTSAM in Motion Model

The **motion model** in this implementation predicts the robot's movement based on odometry, but the odometry data inherently contains noise and drift. By using **GTSAM**, a factor graph is constructed to add constraints between poses over time. These constraints help optimize the robot's trajectory through **pose graph optimization**.

- **Pose Graph**: Each time step is represented as a node in the graph, and odometry is used to create a connection (edge) between consecutive poses. This graph is periodically optimized using **Levenberg-Marquardt** or other optimization techniques to minimize error across the trajectory.
- **Impact on Particle Filter**: GTSAM optimization reduces uncertainty in pose estimates, thus making the subsequent particle filter updates more robust. The particles receive more accurate initial guesses for their poses, allowing the filter to converge faster and build a more reliable map.


## Mapping Process

### Predict Step (Motion Model)
The **motion model** uses odometry data to propagate the particles forward, adding Gaussian noise to account for the uncertainty in the robot's movement. **GTSAM** is used here to maintain an optimized graph of past poses, allowing corrections to the predicted path and reducing drift.

### Update Step (Measurement Model)
The **measurement model** adjusts the weight of each particle based on laser scan data. **ICP** is used to match the predicted scan against the observed scan, resulting in better alignment between the predicted and actual environment.

### Resampling
The **resampling** step is triggered when the **effective number of particles** falls below a threshold. During this step, particles with higher weights are duplicated, while those with lower weights are removed. The result is a new set of particles distributed according to the updated belief about the robot's pose.

## Visualization in RViz
- **Particles Visualization**: The particles are visualized as arrows to show the distribution of possible robot poses. This helps demonstrate the convergence of the particle filter over time.
- **Occupancy Grid Map**: The occupancy grid is published as a **/map** topic and can be visualized in **RViz**. It provides a representation of the environment, with cells marked as occupied, free, or unknown.

## Parameters Used
The following parameters are used in this ROS 2 implementation:

### General Parameters
- **`~throttle_scans`** *(int)*: Skip every nth laser scan.  
- **`~base_frame`** *(string)*: The `tf frame_id` to use for the robot base pose.  
- **`~map_frame`** *(string)*: The `tf frame_id` where the robot pose on the map is published.  
- **`~odom_frame`** *(string)*: The `tf frame_id` from which odometry is read.  
- **`~map_update_interval`** *(double)*: Time in seconds between two recalculations of the map.  

---

### Laser Parameters
- **`~/maxRange`** *(double)*: Maximum range of the laser scans. Rays beyond this range are discarded. *(Default: Maximum laser range - 1 cm)*  
- **`~/maxUrange`** *(double)*: Maximum range of the laser scanner used for map building. *(Default: Same as `maxRange`)*  
- **`~/sigma`** *(double)*: Standard deviation for the scan matching process (cell).  
- **`~/kernelSize`** *(int)*: Search window for the scan matching process.  
- **`~/lstep`** *(double)*: Initial search step for scan matching (linear).  
- **`~/astep`** *(double)*: Initial search step for scan matching (angular).  
- **`~/iterations`** *(int)*: Number of refinement steps in scan matching. The final precision is `lstep * 2^(-iterations)` or `astep * 2^(-iterations)`.  
- **`~/lsigma`** *(double)*: Standard deviation for the scan matching process (single laser beam).  
- **`~/ogain`** *(double)*: Gain for smoothing the likelihood.  
- **`~/lskip`** *(int)*: Use only every `(n+1)`th laser ray for computing a match. *(0 = use all rays)*  
- **`~/minimumScore`** *(double)*: Minimum score for considering the outcome of scan matching good. *(0 = default, Scores go up to 600+)* Example: Try `50` for jumping estimate issues.  

---

### Motion Model Parameters
*(All values represent the standard deviations of a Gaussian noise model.)*
- **`~/srr`** *(double)*: Linear noise component (x and y).  
- **`~/stt`** *(double)*: Angular noise component (theta).  
- **`~/srt`** *(double)*: Linear to angular noise component.  
- **`~/str`** *(double)*: Angular to linear noise component.  

---

### Map Update Parameters
- **`~/linearUpdate`** *(double)*: Process new measurements only if the robot has moved at least this distance (in meters).  
- **`~/angularUpdate`** *(double)*: Process new measurements only if the robot has turned at least this angle (in radians).  
- **`~/resampleThreshold`** *(double)*: Threshold for particle resampling. Higher values result in more frequent resampling.  
- **`~/particles`** *(int)*: Fixed number of particles. Each particle represents a possible trajectory the robot has traveled.  

---

### Likelihood Sampling Parameters
*(Used in scan matching.)*
- **`~/llsamplerange`** *(double)*: Linear sampling range.  
- **`~/lasamplerange`** *(double)*: Angular sampling range.  
- **`~/llsamplestep`** *(double)*: Linear sampling step size.  
- **`~/lasamplestep`** *(double)*: Angular sampling step size.  

---

### Map Dimensions and Resolution
- **`~/xmin`** *(double)*: Minimum x position in the map (in meters).  
- **`~/ymin`** *(double)*: Minimum y position in the map (in meters).  
- **`~/xmax`** *(double)*: Maximum x position in the map (in meters).  
- **`~/ymax`** *(double)*: Maximum y position in the map (in meters).  
- **`~/delta`** *(double)*: Size of one map pixel (in meters).  


## How to Use TurtleBot3 Simulation in Gazebo

To visualize the mapping process using this **ROS 2** implementation, you can simulate **TurtleBot3** in **Gazebo**. Below is a step-by-step guide to set up and use the simulation environment:

### Prerequisites
Ensure that you have installed the following:
- **ROS 2** (Humble/Foxy, or other supported versions).
- **Gazebo** for simulation.
- **TurtleBot3** packages (`turtlebot3`, `turtlebot3_simulations`).

### Step 1: Set Up Environment Variables
To begin, export the environment variables for **TurtleBot3** . In your terminal, run:

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
ros2 run gmapper gmap  
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


or insetead of doing all the steps manually, you can use the following bash script to run all the nodes at the same time. 

```sh
cd <your workspace/src/ros2_gmapping
chmod +x mapping.sh
./mapping.sh
```



Feel free to explore, modify, and extend this implementation for your SLAM projects!
