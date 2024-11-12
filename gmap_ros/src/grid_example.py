#! /usr/bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import TransformStamped, Pose
import tf2_ros
import numpy as np

from scipy.spatial import KDTree

class OccupancyGridMapping(Node):
    def __init__(self):
        super().__init__('occupancy_grid_mapping')
        
        # Initialize parameters and variables
        self.declare_parameter('grid_resolution', 0.05)  # Smaller value means better accuracy
        self.declare_parameter('grid_size', 100)  # Number of grids on each side of the map
        
        # Log-odds parameters for occupancy
        self.declare_parameter('hit_log_odds', 2.0)
        self.declare_parameter('miss_log_odds', -0.5)

        self.resolution = self.get_parameter('grid_resolution').value
        self.grid_size = self.get_parameter('grid_size').value
        self.hit_log_odds = self.get_parameter('hit_log_odds').value  # Occupied 
        self.miss_log_odds = self.get_parameter('miss_log_odds').value  # Free

        # Map origin and initial occupancy grid
        self.origin = [-self.grid_size * self.resolution / 2, -self.grid_size * self.resolution / 2]
        self.log_odds_grid = np.zeros((self.grid_size, self.grid_size), dtype=float)

        # Initialize publishers, subscribers, and broadcaster
        self.occupancy_grid_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Timer to publish occupancy grid and tf at fixed intervals
        self.create_timer(1.0, self.publish_occupancy_grid)
        self.create_timer(0.1, self.publish_map_to_odom_tf)  # Publish transform at 10 Hz

        # Robot pose based on odometry
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0

        # Previous scan for scan matching
        self.previous_scan_points = None

    def odom_callback(self, msg):
        # Update robot pose based on odometry
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_theta = self.get_yaw_from_quaternion(msg.pose.pose.orientation)

    def scan_callback(self, msg):
        # Convert the current laser scan to Cartesian points
        current_scan_points = self.convert_scan_to_points(msg)

        # Perform scan matching if we have a previous scan
        if self.previous_scan_points is not None:
            transformation = self.icp_matching(self.previous_scan_points, current_scan_points)
            # Update the robot pose using the transformation from scan matching
            self.robot_x += transformation[0]
            self.robot_y += transformation[1]
            self.robot_theta += transformation[2]

        # Update the occupancy grid map with the new scan
        for point in current_scan_points:
            self.update_grid_with_scan(point[0], point[1])

        # Save the current scan as the previous scan for the next iteration
        self.previous_scan_points = current_scan_points

    from scipy.spatial import KDTree

    def icp_matching(self, prev_scan_points, curr_scan_points):
        """Simplified ICP implementation using nearest neighbor matching to align scans."""
        prev_points = np.array(prev_scan_points)
        curr_points = np.array(curr_scan_points)

        # Build KDTree for previous points
        tree = KDTree(prev_points)

        # Find nearest neighbors in previous scan for each point in current scan
        distances, indices = tree.query(curr_points)

        # Align the current points to the previous points using nearest neighbors
        matched_points = prev_points[indices]

        # Calculate centroids
        matched_centroid = np.mean(matched_points, axis=0)
        curr_centroid = np.mean(curr_points, axis=0)

        # Center the points
        matched_centered = matched_points - matched_centroid
        curr_centered = curr_points - curr_centroid

        # Compute covariance matrix
        H = np.dot(curr_centered.T, matched_centered)

        # Singular Value Decomposition (SVD)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # Ensure a proper rotation matrix
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Compute translation
        t = matched_centroid - np.dot(R, curr_centroid)

        # Convert rotation matrix to yaw angle
        theta = np.arctan2(R[1, 0], R[0, 0])

        return [t[0], t[1], theta]


    def convert_scan_to_points(self, scan: LaserScan):
        """Convert laser scan ranges to Cartesian coordinates."""
        points = []
        angle = scan.angle_min
        for r in scan.ranges:
            if np.isinf(r) or r <= 0:
                continue
            x = r * np.cos(angle + self.robot_theta) + self.robot_x
            y = r * np.sin(angle + self.robot_theta) + self.robot_y
            points.append((x, y))
            angle += scan.angle_increment
        return points

    def publish_map_to_odom_tf(self):
        # Create a TransformStamped message for the map -> odom transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'

        # Set the transform based on the robot's estimated pose in the map frame
        t.transform.translation.x = self.robot_x
        t.transform.translation.y = self.robot_y
        t.transform.translation.z = 0.0
        t.transform.rotation = self.create_quaternion_from_yaw(self.robot_theta)

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(t)

    def publish_occupancy_grid(self):
        # Clamp log-odds values to prevent overflow
        np.clip(self.log_odds_grid, -10, 10, out=self.log_odds_grid)
        
        # Convert log-odds to probability
        occupancy_data = (1 - 1 / (1 + np.exp(self.log_odds_grid))) * 100  # Convert log-odds to probability
        occupancy_grid_msg = OccupancyGrid()
        occupancy_grid_msg.header.stamp = self.get_clock().now().to_msg()
        occupancy_grid_msg.header.frame_id = 'map'
        occupancy_grid_msg.info.resolution = self.resolution
        occupancy_grid_msg.info.width = self.grid_size
        occupancy_grid_msg.info.height = self.grid_size
        occupancy_grid_msg.info.origin.position.x = self.origin[0]
        occupancy_grid_msg.info.origin.position.y = self.origin[1]
        
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

    def update_grid_with_scan(self, x_laser, y_laser):
        """Update occupancy grid based on the transformed laser point."""
        grid_x, grid_y = self.world_to_grid(x_laser, y_laser)
        robot_grid_x, robot_grid_y = self.world_to_grid(self.robot_x, self.robot_y)
        cells_on_ray = self.bresenham(robot_grid_x, robot_grid_y, grid_x, grid_y)
        for cell_x, cell_y in cells_on_ray:
            if 0 <= cell_x < self.grid_size and 0 <= cell_y < self.grid_size:
                self.log_odds_grid[cell_x, cell_y] += self.miss_log_odds  # Free cell
        if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
            self.log_odds_grid[grid_x, grid_y] += self.hit_log_odds  # Occupied cell

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates."""
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return grid_x, grid_y

    def bresenham(self, x0, y0, x1, y1):
        """Use Bresenham's algorithm to find cells on a line."""
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
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return cells

def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridMapping()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
