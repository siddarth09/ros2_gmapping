#! /usr/bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import TransformStamped
import tf2_ros
import numpy as np

class RBPFSLAMNode(Node):
    def __init__(self):
        super().__init__('rbpf_slam_node')
        
        """
        Parameters used in this SLAM Node:
        
        - throttle_scans: [int] Defines how many laser scans to skip before processing one. Default is 1.
        - base_frame: [string] The tf frame for the robot's base (e.g., 'base_link').
        - map_frame: [string] The tf frame for the map (e.g., 'map').
        - odom_frame: [string] The tf frame for odometry data (e.g., 'odom').
        - map_update_interval: [double] Time interval (in seconds) between two map updates.
        - maxRange: [double] Maximum range of laser scans to be considered. Default is 10 meters.
        - maxUrange: [double] Maximum usable range for map building. Default is the same as maxRange.
        - sigma: [double] Standard deviation for the scan matching process.
        - kernelSize: [int] Search window size for the scan matching process.
        - lstep: [double] Linear step size for scan matching.
        - astep: [double] Angular step size for scan matching.
        - iterations: [int] Number of refinement iterations for scan matching.
        - lsigma: [double] Standard deviation for individual laser beams during scan matching.
        - ogain: [double] Gain for smoothing the likelihood.
        - lskip: [int] Specifies to take every (n+1)th laser ray for matching (0 means take all).
        - minimumScore: [double] Minimum score for considering a scan match successful.
        
        Motion Model Parameters (all represent standard deviations of a Gaussian noise model):
        - srr: [double] Linear noise component (x and y).
        - stt: [double] Angular noise component (theta).
        - srt: [double] Linear to angular noise component.
        - str: [double] Angular to linear noise component.
        
        SLAM Parameters:
        - linearUpdate: [double] Minimum linear movement required to process a new measurement.
        - angularUpdate: [double] Minimum angular movement required to process a new measurement.
        - resampleThreshold: [double] Threshold at which particles get resampled.
        - particles: [int] Number of particles used in the particle filter.
        
        Likelihood Sampling Parameters:
        - llsamplerange: [double] Linear range for likelihood sampling.
        - lasamplerange: [double] Angular range for likelihood sampling.
        - llsamplestep: [double] Linear step size for likelihood sampling.
        - lasamplestep: [double] Angular step size for likelihood sampling.
        
        Map Dimensions and Resolution:
        - xmin, ymin, xmax, ymax: [double] Define the boundaries of the map in meters.
        - delta: [double] Resolution of the map (size of each grid cell in meters).
        """

        # Parameters from the given documentation
        # Parameters related to laser
        self.declare_parameter('throttle_scans', 1)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('map_update_interval', 5.0)
        self.declare_parameter('maxRange', 10.0)
        self.declare_parameter('maxUrange', 10.0)
        self.declare_parameter('sigma', 0.05)
        self.declare_parameter('kernelSize', 1)
        self.declare_parameter('lstep', 0.05)
        self.declare_parameter('astep', 0.05)
        self.declare_parameter('iterations', 5)
        self.declare_parameter('lsigma', 0.075)
        self.declare_parameter('ogain', 3.0)
        self.declare_parameter('lskip', 0)
        self.declare_parameter('minimumScore', 50.0)
        
        # Motion model parameters
        self.declare_parameter('srr', 0.1)
        self.declare_parameter('stt', 0.1)
        self.declare_parameter('srt', 0.1)
        self.declare_parameter('str', 0.1)
        
        # SLAM parameters
        self.declare_parameter('linearUpdate', 0.5)
        self.declare_parameter('angularUpdate', 0.2)
        self.declare_parameter('resampleThreshold', 0.5)
        self.declare_parameter('particles', 30)
        
        # Likelihood sampling parameters
        self.declare_parameter('llsamplerange', 0.1)
        self.declare_parameter('lasamplerange', 0.1)
        self.declare_parameter('llsamplestep', 0.05)
        self.declare_parameter('lasamplestep', 0.05)
        
        # Map dimensions
        self.declare_parameter('xmin', -10.0)
        self.declare_parameter('ymin', -10.0)
        self.declare_parameter('xmax', 10.0)
        self.declare_parameter('ymax', 10.0)
        self.declare_parameter('delta', 0.05)
        
        # Load parameters
        self.load_parameters()
        
        #publishers, subscribers, and TF broadcaster
        self.occupancy_grid_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Timer for publishing the map
        self.create_timer(self.map_update_interval, self.publish_occupancy_grid)
        self.create_timer(0.1, self.publish_map_to_odom_tf)  # Publish TF at 10 Hz

        # Variables to store robot pose and map data
        self.robot_pose = [0.0, 0.0, 0.0]
        self.log_odds_grid = np.zeros((int((self.xmax - self.xmin) / self.delta),
                                       int((self.ymax - self.ymin) / self.delta)))
        self.previous_scan_points = None

    def load_parameters(self):
        # Load parameters for use in the node
        self.throttle_scans = self.get_parameter('throttle_scans').value
        self.base_frame = self.get_parameter('base_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.map_update_interval = self.get_parameter('map_update_interval').value
        self.max_range = self.get_parameter('maxRange').value
        self.max_urange = self.get_parameter('maxUrange').value
        self.sigma = self.get_parameter('sigma').value
        self.kernel_size = self.get_parameter('kernelSize').value
        self.lstep = self.get_parameter('lstep').value
        self.astep = self.get_parameter('astep').value
        self.iterations = self.get_parameter('iterations').value
        self.lsigma = self.get_parameter('lsigma').value
        self.ogain = self.get_parameter('ogain').value
        self.lskip = self.get_parameter('lskip').value
        self.minimum_score = self.get_parameter('minimumScore').value
        
        self.srr = self.get_parameter('srr').value
        self.stt = self.get_parameter('stt').value
        self.srt = self.get_parameter('srt').value
        self.str = self.get_parameter('str').value
        
        self.linear_update = self.get_parameter('linearUpdate').value
        self.angular_update = self.get_parameter('angularUpdate').value
        self.resample_threshold = self.get_parameter('resampleThreshold').value
        self.particles_count = self.get_parameter('particles').value
        
        self.llsamplerange = self.get_parameter('llsamplerange').value
        self.lasamplestep = self.get_parameter('lasamplerange').value
        self.llsamplestep = self.get_parameter('llsamplestep').value
        self.lasamplestep = self.get_parameter('lasamplestep').value
        
        self.xmin = self.get_parameter('xmin').value
        self.ymin = self.get_parameter('ymin').value
        self.xmax = self.get_parameter('xmax').value
        self.ymax = self.get_parameter('ymax').value
        self.delta = self.get_parameter('delta').value

    def odom_callback(self, msg):
        # Update robot pose based on odometry
        self.robot_pose[0] = msg.pose.pose.position.x
        self.robot_pose[1] = msg.pose.pose.position.y
        self.robot_pose[2] = self.get_yaw_from_quaternion(msg.pose.pose.orientation)

    def scan_callback(self, msg):
        
        curr_scan_points = []
        angle = msg.angle_min
        for r in msg.ranges:
            if 0 < r < self.max_range:
                x = self.robot_pose[0] + r * np.cos(angle + self.robot_pose[2])
                y = self.robot_pose[1] + r * np.sin(angle + self.robot_pose[2])
                curr_scan_points.append([x, y])
            angle += msg.angle_increment
        curr_scan_points = np.array(curr_scan_points)

        if self.previous_scan_points is not None:
            # Likelihood Computation:
            transformation, score = self.icp_matching(self.previous_scan_points, curr_scan_points)
            if score > self.minimum_score:
                # Pose Optimization: Adjust the pose to maximize the alignment likelihood
                self.robot_pose[0] += transformation[0, 2]
                self.robot_pose[1] += transformation[1, 2]
                self.robot_pose[2] += np.arctan2(transformation[1, 0], transformation[0, 0])

        self.previous_scan_points = curr_scan_points

    def icp_matching(self, prev_points, curr_points):
        # ICP (Iterative Closest Point) matching
        prev_center = np.mean(prev_points, axis=0)
        curr_center = np.mean(curr_points, axis=0)

        prev_centered = prev_points - prev_center
        curr_centered = curr_points - curr_center

        H = np.dot(curr_centered.T, prev_centered)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        t = prev_center.T - np.dot(R, curr_center.T)

        transformation = np.identity(3)
        transformation[:2, :2] = R
        transformation[:2, 2] = t

        # Compute score based on the alignment (simple distance measure)
        aligned_points = np.dot(curr_centered, R.T) + t
        score = -np.sum(np.linalg.norm(prev_centered - aligned_points, axis=1))

        return transformation, score

    def publish_map_to_odom_tf(self):
        #TransformStamped message for the map -> odom transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame
        t.transform.translation.x = self.robot_pose[0]
        t.transform.translation.y = self.robot_pose[1]
        t.transform.translation.z = 0.0
        t.transform.rotation = self.create_quaternion_from_yaw(self.robot_pose[2])
        
        self.tf_broadcaster.sendTransform(t)

    def publish_occupancy_grid(self):
       
        occupancy_grid_msg = OccupancyGrid()
        occupancy_grid_msg.header.stamp = self.get_clock().now().to_msg()
        occupancy_grid_msg.header.frame_id = self.map_frame
        occupancy_grid_msg.info.resolution = self.delta
        occupancy_grid_msg.info.width = self.log_odds_grid.shape[1]
        occupancy_grid_msg.info.height = self.log_odds_grid.shape[0]
        occupancy_grid_msg.info.origin.position.x = self.xmin
        occupancy_grid_msg.info.origin.position.y = self.ymin
        
        occupancy_data = (1 - 1 / (1 + np.exp(self.log_odds_grid))) * 100  # Convert log-odds to probability
        occupancy_grid_msg.data = occupancy_data.flatten().astype(int).tolist()
        self.occupancy_grid_pub.publish(occupancy_grid_msg)

    def get_yaw_from_quaternion(self, orientation):
        """Convert a quaternion into yaw angle."""
        qx = orientation.x
        qy = orientation.y
        qz = orientation.z
        qw = orientation.w
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        return np.arctan2(siny_cosp, cosy_cosp)

    def create_quaternion_from_yaw(self, yaw):
        """Helper function to create a quaternion from a yaw angle."""
        from geometry_msgs.msg import Quaternion
        q = Quaternion()
        q.z = np.sin(yaw / 2.0)
        q.w = np.cos(yaw / 2.0)
        return q

def main(args=None):
    rclpy.init(args=args)
    node = RBPFSLAMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
