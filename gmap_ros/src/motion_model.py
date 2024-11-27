#! /usr/bin/env python3 

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
import gtsam
from gtsam import Pose2, PriorFactorPose2, BetweenFactorPose2, Values, NonlinearFactorGraph
from gtsam.symbol_shorthand import X
from tf_transformations import quaternion_from_euler,euler_from_quaternion
import tf2_ros

class MotionModel(Node):
    def __init__(self):
        super().__init__('motion_model_gtsam')

        self.declare_parameter("particles", 1000)
        self.num_particles = self.get_parameter("particles").value 
        self.particles = [[0.0, 0.0, 0.0] for _ in range(self.num_particles)]
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.tf_broadcaster=tf2_ros.TransformBroadcaster(self)
        # Publisher for the optimized pose
        self.motion_model_pub = self.create_publisher(PoseStamped, '/motion_model', 10)

        # GTSAM variables
        self.graph = NonlinearFactorGraph()  
        self.initial_estimate = Values()     
        self.previous_pose = Pose2(0.0, 0.0, 0.0)  # Initialize previous pose
        self.odom_covariance = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1])) 

        # Adding prior factor for the initial pose
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))  
        self.graph.add(PriorFactorPose2(X(0), self.previous_pose, prior_noise))
        self.initial_estimate.insert(X(0), self.previous_pose)

        self.odom_count = 0
        self.prev_time=None 
    
    def odom_callback(self,msg):
        twist_covariance = np.array(msg.twist.covariance).reshape((6, 6))
        
        q = (
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w)
        
        # self.get_logger().warn(f"euler values:{q}")
        quaterion=euler_from_quaternion(q)
        self.get_logger().warn(f"euler values:{quaterion}")
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.previous_pose=Pose2(msg.pose.pose.position.x,
                                 msg.pose.pose.position.y,
                                 quaterion[2]         
        )
        linear_x_variance = twist_covariance[0, 0]
        linear_y_variance = twist_covariance[1, 1]
        angular_z_variance = twist_covariance[5, 5]
        
        if self.prev_time is None: 
            DT=0.1 
        else: 
            DT= current_time-self.prev_time
        self.prev_time=current_time
        
        
        if DT<=0.0: 
            self.get_logger().warn("Non positive DT is detected")
            
            
        F = np.array([[1.0, 0, 0],
                      [0, 1.0, 0],
                      [0, 0, 1.0]])

        # Loop through all particles and apply motion update with noise
        for i in range(self.num_particles):
            # Current particle state
            x = self.particles[i]

            # Generate noisy control inputs
            dx = np.random.normal(msg.twist.twist.linear.x, np.sqrt(linear_x_variance))
            dy = np.random.normal(msg.twist.twist.linear.y, np.sqrt(linear_y_variance))     
            dtheta = np.random.normal(msg.twist.twist.angular.z, np.sqrt(angular_z_variance))

            # Convert odometry data from centimeters to meters
            dx /= 100.0
            dy /= 100.0

            # Define B matrix using the noisy inputs and current orientation
            B = np.array([[DT * np.cos(x[2]), 0],
                          [DT * np.sin(x[2]), 0],
                          [0.0, DT]])

            # Update the particle position using F and B
            u = np.array([[dx, dtheta]]).T
            x = F @ x + B @ u
            self.particles[i] = [x[0], x[1], x[2]]

        # Get the previous pose from the current state of the robot
        avg_x = np.mean([p[0] for p in self.particles])
        avg_y = np.mean([p[1] for p in self.particles])
        avg_theta = np.mean([p[2] for p in self.particles])
        self.previous_pose = Pose2(avg_x, avg_y, avg_theta)

        # Proceed with the GTSAM graph update as before
        odom_dx = msg.twist.twist.linear.x / 100.0  # Convert to meters
        odom_dy = msg.twist.twist.linear.y / 100.0  # Convert to meters
        odom_dtheta = msg.twist.twist.angular.z
        odom_delta = Pose2(odom_dx, odom_dy, odom_dtheta)

        current_pose_key = X(self.odom_count + 1)
        previous_pose_key = X(self.odom_count)

        self.graph.add(BetweenFactorPose2(previous_pose_key, current_pose_key, odom_delta, self.odom_covariance))
        
        # Initial estimate for the current pose
        new_estimate = self.previous_pose.compose(odom_delta)
        self.initial_estimate.insert(current_pose_key, new_estimate)

        # Optimize the factor graph using GTSAM
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate)
        result = optimizer.optimize()

        # Retrieve the optimized pose
        self.previous_pose = result.atPose2(current_pose_key)
        optimized_pose = [self.previous_pose.x(), self.previous_pose.y(), self.previous_pose.theta()]

        # Publish the optimized pose to the /motion_model topic
        self.publish_optimized_pose(optimized_pose)

        # Update particles based on GTSAM optimized pose
        for i in range(self.num_particles):
            # Set particle positions directly from the optimized pose with added Gaussian noise
            noisy_x = np.random.normal(optimized_pose[0], 0.1)
            noisy_y = np.random.normal(optimized_pose[1], 0.1)
            noisy_theta = np.random.normal(optimized_pose[2], 0.05)
            self.particles[i] = [noisy_x, noisy_y, noisy_theta]

        # Increment odometry count
        self.odom_count += 1

        # Log the optimized pose
        self.get_logger().info(f"Optimized Pose: {optimized_pose}")        

    def publish_optimized_pose(self, optimized_pose):

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"  
        pose_msg.pose.position.x = optimized_pose[0]
        pose_msg.pose.position.y = optimized_pose[1]
        pose_msg.pose.position.z = 0.0

        q = quaternion_from_euler(0.0, 0.0, optimized_pose[2])
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]

        self.motion_model_pub.publish(pose_msg)

       
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'
        t.transform.translation.x = optimized_pose[0]
        t.transform.translation.y = optimized_pose[1]
        t.transform.translation.z = 0.0
        t.transform.rotation = pose_msg.pose.orientation

        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    motion_model_node = MotionModel()
    rclpy.spin(motion_model_node)
    motion_model_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
