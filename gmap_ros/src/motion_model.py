#! /usr/bin/env python3 

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import gtsam
from gtsam import Pose2, PriorFactorPose2, BetweenFactorPose2, Values, NonlinearFactorGraph
from gtsam.symbol_shorthand import X

class MotionModel(Node):
    def __init__(self):
        super().__init__('motion_model_gtsam')

        
        self.declare_parameter("particles", 1000)
        self.num_particles = self.get_parameter("particles").value 
        self.particles = [[0.0, 0.0, 0.0] for _ in range(self.num_particles)]
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # GTSAM variables
        self.graph = NonlinearFactorGraph()  
        self.initial_estimate = Values()     
        self.previous_pose = Pose2(0.0, 0.0, 0.0)  
        self.odom_covariance = gtsam.noiseModel_Diagonal.Sigmas(np.array([0.1, 0.1, 0.1])) 

        # Adding prior factor for the initial pose
        prior_noise = gtsam.noiseModel_Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))  
        self.graph.add(PriorFactorPose2(X(0), self.previous_pose, prior_noise))
        self.initial_estimate.insert(X(0), self.previous_pose)

        
        self.odom_count = 0

    def odom_callback(self, msg):
        twist_covariance = np.array(msg.twist.covariance).reshape((6, 6))
        
        linear_x_variance = twist_covariance[0, 0]
        linear_y_variance = twist_covariance[1, 1]
        angular_z_variance = twist_covariance[5, 5]

        # Loop through all particles and apply motion update with noise
        for i in range(self.num_particles):
            
            dx = np.random.normal(msg.twist.twist.linear.x, np.sqrt(linear_x_variance))
            dy = np.random.normal(msg.twist.twist.linear.y, np.sqrt(linear_y_variance))     
            dtheta = np.random.normal(msg.twist.twist.angular.z, np.sqrt(angular_z_variance))

            # Update the particle position
            self.particles[i][0] += dx
            self.particles[i][1] += dy
            self.particles[i][2] += dtheta

        
        odom_dx = msg.twist.twist.linear.x
        odom_dy = msg.twist.twist.linear.y
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

        # Update particles based on GTSAM optimized pose
        for i in range(self.num_particles):
            # Adding Gaussian noise around the optimized pose for particles
            noisy_x = np.random.normal(optimized_pose[0], 0.1)
            noisy_y = np.random.normal(optimized_pose[1], 0.1)
            noisy_theta = np.random.normal(optimized_pose[2], 0.05)
            self.particles[i] = [noisy_x, noisy_y, noisy_theta]

        # Increment odometry count
        self.odom_count += 1

        # Log the optimized pose and updated particles
        self.get_logger().info(f"Optimized Pose: {optimized_pose}")
        self.get_logger().info(f"Updated particles: {self.particles}")

    def get_proposal_dist(self):
        return self.particles

    def dead_reckoning(self):
        # TODO: to build a dead reckoning model
        pass 

def main():
    rclpy.init()
    motion_model_node = MotionModel()
    rclpy.spin(motion_model_node)
    motion_model_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
