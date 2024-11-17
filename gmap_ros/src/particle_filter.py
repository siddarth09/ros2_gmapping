#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, Quaternion,TransformStamped
import numpy as np
import random
from motion_model import MotionModel
from measurement_model import MeasurementModel
import tf2_ros

class ParticleFilterMappingNode(Node):
    def __init__(self):
        super().__init__('rbpf_particle_filter_mapping')

        # Parameters for Particle Filter
        self.declare_parameter('particles_count', 1000)
        self.num_particles = self.get_parameter('particles_count').value

        # Parameters for Occupancy Grid Mapping
        self.declare_parameter('grid_resolution', 0.05)  # Cell size in meters
        self.declare_parameter('grid_size', 100)  # Number of cells in each dimension

        # Initialize particles: each particle is [x, y, theta] with weight
        self.particles = [[0.0, 0.0, 0.0] for _ in range(self.num_particles)]
        self.weights = [1.0 / self.num_particles] * self.num_particles

        # Create instances of MotionModel and MeasurementModel
        self.motion_model = MotionModel()
        self.measurement_model = MeasurementModel()

        # Subscribe to Odometry and LaserScan topics
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.predict, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.observation_update, 10)

        # Publisher for visualizing particles and occupancy grid map in RViz
        self.particles_pub = self.create_publisher(PoseArray, '/particles', 10)
        self.occupancy_grid_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_timer= self.create_timer(0.1, self.publish_map_to_odom_tf)
        self.grid_timer=self.create_timer(0.1,self.publish_occupancy_grid)
        # Occupancy Grid Map initialization
        self.grid_resolution = self.get_parameter('grid_resolution').value
        self.grid_size = self.get_parameter('grid_size').value
        self.origin = [-self.grid_size * self.grid_resolution / 2, -self.grid_size * self.grid_resolution / 2]
        self.log_odds_grid = np.zeros((self.grid_size, self.grid_size), dtype=float)

        # Laser scan data for update
        self.last_scan_points = None

    def predict(self, msg):
        # Predict Step: Update each particle's pose based on the odometry data using the motion model
        self.motion_model.odom_callback(msg)
        updated_particles = self.motion_model.get_proposal_dist()

        # Update particles with the new proposal distribution
        for i in range(self.num_particles):
            self.particles[i][0] = updated_particles[i][0]
            self.particles[i][1] = updated_particles[i][1]
            self.particles[i][2] = updated_particles[i][2]

    def observation_update(self, msg):
        # Update Step: Adjust the weights of each particle using the measurement model
        self.measurement_model.last_odom = self.motion_model.previous_pose
        self.measurement_model.scan_callback(msg)
        
        #From measurement model after ICP
        updated_particles = self.measurement_model.particles 
        updated_weights = self.measurement_model.weights  

        
        self.particles = updated_particles
        self.weights = updated_weights

        self.normalize_weights()
        
        if self.effective_particle_count() < self.num_particles / 2:
            self.resample_particles()

        # Update the occupancy grid map using the particles and scan data
        self.update_occupancy_grid(self.measurement_model.last_scan_points)

        # Publish the particle poses for visualization
        self.publish_particles()

        # Publish the occupancy grid map for visualization
        self.publish_occupancy_grid()

    def normalize_weights(self):
       
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]

    def effective_particle_count(self):
        
        weights = np.array(self.weights)
        return 1.0 / np.sum(weights ** 2)

    def resample_particles(self):
        # Resample particles based on the weights
        new_particles = []
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  
        step = 1.0 / self.num_particles
        r = random.uniform(0, step)
        index = 0

        for _ in range(self.num_particles):
            while r > cumulative_sum[index]:
                index += 1
            new_particles.append(self.particles[index][:]) 
            r += step

        # Replace particles with the resampled ones and reset weights
        self.particles = new_particles
        self.weights = [1.0 / self.num_particles] * self.num_particles

    def update_occupancy_grid(self, scan_points):
        
        if scan_points is None:
            return

        # Get particle with the highest weight (most probable pose)
        best_p_index = np.argmax(self.weights)
        best_p = self.particles[best_p_index]

        robot_grid_x, robot_grid_y = self.world_to_grid(best_p[0], best_p[1])

        for point in scan_points:
            # Convert laser scan points to occupancy grid cells
            grid_x, grid_y = self.world_to_grid(point[0], point[1])

            # Use Bresenham's algorithm to update the cells along the ray
            cells_on_ray = self.bresenham(robot_grid_x, robot_grid_y, grid_x, grid_y)

            # Update log odds for free cells
            for cell_x, cell_y in cells_on_ray:
                if 0 <= cell_x < self.grid_size and 0 <= cell_y < self.grid_size:
                    self.log_odds_grid[cell_x, cell_y] -= 0.1  # Miss log-odds (free space)

            # Update log odds for the occupied cell (end of laser scan)
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.log_odds_grid[grid_x, grid_y] += 0.7  # Hit log-odds (occupied space)

    def world_to_grid(self, x, y):
        # Convert world coordinates to grid indices
        grid_x = int((x - self.origin[0]) / self.grid_resolution)
        grid_y = int((y - self.origin[1]) / self.grid_resolution)
        return grid_x, grid_y

    def bresenham(self, x0, y0, x1, y1):
        # Implementation of Bresenham's line algorithm to determine the cells along a ray
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return cells

    def publish_particles(self):
        # Publish particles as PoseArray for RViz visualization
        particles_msg = PoseArray()
        particles_msg.header.frame_id = 'map'
        particles_msg.header.stamp = self.get_clock().now().to_msg()

        for particle in self.particles:
            pose = Pose()
            pose.position.x = particle[0]
            pose.position.y = particle[1]
            quaternion = self.create_quaternion_from_yaw(particle[2])
            pose.orientation = quaternion
            particles_msg.poses.append(pose)

        self.particles_pub.publish(particles_msg)

    def publish_occupancy_grid(self):
        # Converting the log-odds values to probabilities and publish the occupancy grid map
        
        
        # Updating the clip function from octomap 
        np.clip(self.log_odds_grid, -10, 10, out=self.log_odds_grid)
        occupancy_data = (1 - 1 / (1 + np.exp(self.log_odds_grid))) * 100  

        occupancy_grid_msg = OccupancyGrid()
        occupancy_grid_msg.header.stamp = self.get_clock().now().to_msg()
        occupancy_grid_msg.header.frame_id = 'map'
        occupancy_grid_msg.info.resolution = self.grid_resolution
        occupancy_grid_msg.info.width = self.grid_size
        occupancy_grid_msg.info.height = self.grid_size
        occupancy_grid_msg.info.origin.position.x = self.origin[0]
        occupancy_grid_msg.info.origin.position.y = self.origin[1]
        occupancy_grid_msg.data = occupancy_data.flatten().astype(int).tolist()

        self.occupancy_grid_pub.publish(occupancy_grid_msg)
        
    # def publish_map_to_odom_tf(self):
    #     # Publish the transform between the map and odom frames using the robot's most probable pose
    #     best_p_index = np.argmax(self.weights)
    #     best_p = self.particles[best_p_index]

    #     t = TransformStamped()
    #     t.header.stamp = self.get_clock().now().to_msg()
    #     t.header.frame_id = 'map'
    #     t.child_frame_id = 'odom'
    #     t.transform.translation.x = best_p[0]
    #     t.transform.translation.y = best_p[1]
    #     t.transform.translation.z = 0.0
    #     q = self.create_quaternion_from_yaw(best_p[2])
    #     t.transform.rotation = q

    #     self.tf_broadcaster.sendTransform(t)
        
    def publish_map_to_odom_tf(self):
        # Publish the transform between the map and odom frames using the robot's most probable pose
        best_p_index = np.argmax(self.weights)
        best_p = self.particles[best_p_index]

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        q = self.create_quaternion_from_yaw(best_p[2])
        t.transform.rotation = q

        self.tf_broadcaster.sendTransform(t)

    def create_quaternion_from_yaw(self, yaw):
        # Creating a quaternion from a given yaw angle
        q = Quaternion()
        q.z = np.sin(yaw / 2.0)
        q.w = np.cos(yaw / 2.0)
        return q

def main(args=None):
    rclpy.init(args=args)
    particle_filter_node = ParticleFilterMappingNode()
    rclpy.spin(particle_filter_node)
    particle_filter_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
