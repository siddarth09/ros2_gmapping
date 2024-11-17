#! /usr/bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, Quaternion
from nav_msgs.msg import Odometry
import numpy as np
import random
from scipy.spatial import KDTree
import gtsam
from gtsam import Pose2, Values, NonlinearFactorGraph, BetweenFactorPose2, noiseModel

class MeasurementModel(Node):
    def __init__(self):
        super().__init__('measurement_model')
        
        # Declare parameters
        self.declare_parameter('particles_count', 1000)
        self.particles_count = self.get_parameter('particles_count').value


        self.particles = [[0.0, 0.0, 0.0] for _ in range(self.particles_count)]
        self.weights = [1.0 / self.particles_count] * self.particles_count

        # GTSAM variables for graph optimization
        self.graph = NonlinearFactorGraph()
        self.initial_estimate = Values()
        self.optimized_values = Values()
        self.pose_noise = noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  

        # Subscribers to the laser scan and odometry topics
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Publisher for visualizing particles in RViz
        self.particles_pub = self.create_publisher(PoseArray, '/particles', 10)
        self.timer=self.create_timer(0.1,self.publish_particles)

        # Storage for odometry and scan data
        self.last_odom = None
        self.last_scan_points = None
        self.previous_pose = Pose2(0.0, 0.0, 0.0)
        self.current_pose_index = 0

    def odom_callback(self, msg):
        self.last_odom = msg

    def scan_callback(self, msg):
        if not self.last_odom:
            
            return

        
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        range_min = msg.range_min
        range_max = msg.range_max
        ranges = msg.ranges

        # Convert LaserScan ranges to Cartesian coordinates (x, y)
        scan_points = []
        angle = angle_min
        for r in ranges:
            if range_min < r < range_max:
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                scan_points.append([x, y])
            
            angle += angle_increment

       
        scan_points = np.array(scan_points)

       #ICP
        for i in range(self.particles_count):
            particle = self.particles[i]

            if self.last_scan_points is not None:
                
                transformation = self.icp_matching(self.last_scan_points, scan_points)

                
                particle[0] += transformation[0, 2]  # x translation
                particle[1] += transformation[1, 2]  # y translation
                delta_theta = np.arctan2(transformation[1, 0], transformation[0, 0])
                particle[2] += delta_theta  # Update orientation

                #Weight Update
                self.weights[i] *= self.compute_likelihood(scan_points, self.transform_points(self.last_scan_points, transformation))

        # Normalize weights
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]

        self.publish_particles()
        self.last_scan_points = scan_points

    def icp_matching(self, prev_points, curr_points):
        # Use KDTree to find the closest points in the current scan to the previous points
        kdtree = KDTree(curr_points)
        distances, indices = kdtree.query(prev_points)

        # Compute centroids of the point sets
        prev_centroid = np.mean(prev_points, axis=0)
        curr_centroid = np.mean(curr_points[indices.flatten()], axis=0)

        #Find centroids
        prev_centered = prev_points - prev_centroid
        curr_centered = curr_points[indices.flatten()] - curr_centroid

        # Compute the covariance matrix
        H = np.dot(curr_centered.T, prev_centered)

        # Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(U, Vt)
        t = curr_centroid.T - np.dot(R, prev_centroid.T)

        # Construct the transformation matrix
        transformation = np.identity(3)
        transformation[0:2, 0:2] = R
        transformation[0:2, 2] = t

        return transformation

    def compute_likelihood(self, scan1, scan2):
        # Find the minimum length of both scans
        min_length = min(len(scan1), len(scan2))
        
        # Trim both scans to the same size
        trimmed_scan1 = scan1[:min_length]
        trimmed_scan2 = scan2[:min_length]
        
        # Compute the difference between two sets of points and return a likelihood
        diff = np.linalg.norm(trimmed_scan1 - trimmed_scan2, axis=1)
        return np.exp(-np.sum(diff) / len(trimmed_scan1))


    def transform_points(self, points, transformation):
        
        transformed_points = []
        for point in points:
            x, y = point
            transformed = np.dot(transformation, np.array([x, y, 1]))
            transformed_points.append([transformed[0], transformed[1]])
        return np.array(transformed_points)

    def publish_particles(self):
        # Publish particles for RViz visualization
        particles_msg = PoseArray()
        particles_msg.header.frame_id = 'odom'
        particles_msg.header.stamp = self.get_clock().now().to_msg()

        for particle in self.particles:
            pose = Pose()
            pose.position.x = particle[0]
            pose.position.y = particle[1]
           
            quaternion = self.create_quaternion_from_yaw(particle[2])
            pose.orientation = quaternion
            particles_msg.poses.append(pose)

        self.particles_pub.publish(particles_msg)

    def create_quaternion_from_yaw(self, yaw):
        #Quaternion from yaw
        q = Quaternion()
        q.z = np.sin(yaw / 2.0)
        q.w = np.cos(yaw / 2.0)
        return q

    def update_pose_graph(self):
        # Convert the current particle with highest weight to a GTSAM pose
        best_particle_idx = np.argmax(self.weights)
        best_particle = self.particles[best_particle_idx]
        current_pose = Pose2(best_particle[0], best_particle[1], best_particle[2])

        # Add a factor to the graph to connect the previous pose to the current pose
        self.graph.add(BetweenFactorPose2(self.current_pose_index, self.current_pose_index + 1, current_pose.between(self.previous_pose), self.pose_noise))

        # Add the current pose to the initial estimates
        self.initial_estimate.insert(self.current_pose_index + 1, current_pose)

        # Update previous pose and pose index
        self.previous_pose = current_pose
        self.current_pose_index += 1

        print(self.graph)
        # Optimize the pose graph
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate)
        self.optimized_values = optimizer.optimize()
        self.get_logger().info(f"Pose optimization results : \n{self.optimized_values}")
        # Update particles to reflect optimized pose
        optimized_pose = self.optimized_values.atPose2(self.current_pose_index)
        for i in range(self.particles_count):
            self.particles[i][0] += (optimized_pose.x() - best_particle[0])
            self.particles[i][1] += (optimized_pose.y() - best_particle[1])
            self.particles[i][2] += (optimized_pose.theta() - best_particle[2])
        return self.particles

def main(args=None):
    rclpy.init(args=args)
    node = MeasurementModel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
