import numpy as np
from random import randrange, choice
import math
import matplotlib.pyplot as plt
import copy
import time
from scipy import signal, ndimage
from skimage.draw import line       # For ray tracing

import rospy
import tf2_ros
import tf2_geometry_msgs
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry, Path
from std_msgs.msg import Float64, Header, ColorRGBA
from geometry_msgs.msg import Point, PointStamped, Pose, Twist, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import actionlib
import arl_nav_msgs.msg as nav_msg
from actionlib_msgs.msg import GoalID
from sensor_msgs.msg import LaserScan, PointCloud2
import sensor_msgs.point_cloud2 as pc2

from itertools import chain
import struct
import pyastar2d                        # fast A* path planning
import csv
import os
import datetime
import gtsam
from threading import Lock


class graph_active_slam:
    def __init__(self):

        rospy.init_node('graph_active_slam')
        rospy.sleep(10.)

        rospy.loginfo("GraphBasedSLAM node started")

        self.output_file = rospy.get_param("output_file", "/home/arl-sse/phoenix-r1-copy/src/behaviors/single_robot_infosplorer/time_series_data_Exp4_30_v2_Renyi.csv")

        self.cnt = 0
        self.sim_env = 2    # Simulation evirnoment - 1: 8 rooms / 2: 32 rooms / 3: uc1

        if self.sim_env == 1:
            self.offset_x = 0.0
            self.offset_y = 0.0
        elif self.sim_env == 2:
            self.offset_x = 0.0
            self.offset_y = -18.0
        elif self.sim_env == 3:
            self.offset_x = -121.0
            self.offset_y = -40.0

        self.lidar_range_noise = 0.01**2 
        # self.R_odom = np.diag([0.1, 0.1, np.deg2rad(0.02)]) ** 2              # odometry noise covariance
        self.R_odom = np.diag([0.01, 0.01, np.deg2rad(0.06)]) ** 2              # Low
        # self.R_odom = np.diag([0.05, 0.05, np.deg2rad(0.3)]) ** 2             # Medium
        # self.R_odom = np.diag([0.1, 0.1, np.deg2rad(1.2)]) ** 2               # High
        # self.R_odom = np.diag([0.1, 0.1, np.deg2rad(3.0)]) ** 2               # Much High

        self.graph = None
        self.current_pose_id = 0
        self.keyframes = []
        self.keyframe_poses = []
        self.keyframe_scans = []
        self.keyframe_poses_cov = []
        self.keyframe_ground_truth = []
        self.latest_odom = None
        self.latest_scan = None
        self.latest_transform = None
        self.path_x_coord = np.array([])
        self.path_y_coord = np.array([])

        # Parameters for occupancy grid
        self.accumulated_cloud = None
        self.resolution = 0.2           # It will get from cost_map_cb
        self.map_update_frequency = 3   # Update map every 3 keyframes (to reduce computation)
        
        self.costmap_initialized = False

        # Parameters for gtsam
        self.keyframe_distance = 0.5  # Min distance between keyframes
        self.keyframe_angle = 0.2     # Min angle between keyframes
        self.loop_closure_threshold = 0.7  # ICP fitness threshold for loop closure
        self.scan_matching_max_iter = 30
        self.scan_matching_resolution = 0.1
        self.max_range = 20.0  # Max sensor range

        # Parameters for log-odd map
        self.log_odds_occupied = 1.5    # Slightly reduced from 1.0
        self.log_odds_free = -1.0       # Slightly reduced from -0.7
        self.log_odds_prior = 0.0
        self.log_odds_max = 5.0
        self.log_odds_min = -5.0
        self.log_odds_threshold = 1.5   # Threshold for updating cell probability
        self.reupdate_factor = 1.5      # Re-update factor for cells with low uncertainty(1.0 is default)
        self.log_odds_re_update = False

        buffer_duration = rospy.Duration(60.0)
        self.tf_buffer = tf2_ros.Buffer(buffer_duration)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Initialize GTSAM
        self.initialize_gtsam()

        ''' Navigation related parameters'''
        self.goal_tol=3.0 # Distance (m) to switch to next goal
        self.goal_radius=6.0 # Goal radius in (m) # was 4 initially
        self.goal_angle_thresh = 7.0 # Goal Angle threshold in radians 
        self.recovery_time= 4.0 # time in seconds to start recovery behavior
        self.last_marker_time=time.time()
        self.execution_time=rospy.get_param("/execution_time")

        self.goal_cnt=0

        self.reached_goal = True
        
        # Publishers
        self.pose_publisher = rospy.Publisher('/graph_slam/pose', PoseStamped, queue_size=10)
        self.path_publisher = rospy.Publisher('/graph_slam/path', Marker, queue_size=10)
        self.viz_publisher = rospy.Publisher('/graph_slam/visualization', MarkerArray, queue_size=10)
        self.map_publisher = rospy.Publisher('/graph_slam/occupancy_grid', OccupancyGrid, queue_size=1)
        self.loop_pub = rospy.Publisher('/graph_slam/loop_closures', Marker, queue_size=10)
        self.vel_pub = rospy.Publisher('/warty/warthog_velocity_controller/cmd_vel', Twist, queue_size=10, latch=True)

        # Subscribers
        rospy.Subscriber("/warty/odom", Odometry, self.odom_cb, queue_size=10)
        rospy.Subscriber('/warty/lidar_points_center', PointCloud2, self.scan_cb, queue_size=10)
        rospy.Subscriber("frontiers_cluster", Marker, self.marker_cb, queue_size=1) # Get latest messsage
        rospy.Subscriber("/unity_command/ground_truth/warty/pose", PoseStamped, self.ground_truth_cb, queue_size=10)
        rospy.Subscriber("/warty/global_costmap/costmap/costmap", OccupancyGrid, self.costmap_cb, queue_size=10)
        rospy.Subscriber("/graph_slam/path", Marker, self.path_cb, queue_size=1) 
        
        # Thread lock for data safety
        self.lock = Lock()

        self.rate = rospy.Rate(1) # Run every 10 seconds

        rospy.spin()


    def initialize_gtsam(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.current_estimate = gtsam.Values()
        
        # Add a prior on the first pose
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.01]))
        self.graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(0, 0, 0), prior_noise))
        self.initial_estimate.insert(0, gtsam.Pose2(0, 0, 0))
        
        # Create noise models
        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([np.sqrt(self.R_odom[0, 0]),
                                                                    np.sqrt(self.R_odom[1, 1]),
                                                                    np.sqrt(self.R_odom[2, 2])]))
        
        # self.loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))

        # Create robust loop closure noise model with Huber loss
        # base_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, np.deg2rad(10.0)]))
        base_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, np.deg2rad(5.0)]))
        self.loop_noise = gtsam.noiseModel.Robust.Create(
            # gtsam.noiseModel.mEstimator.Huber.Create(1.345),
            gtsam.noiseModel.mEstimator.Cauchy.Create(0.5),  # Even more robust than Huber
            base_noise
        )
        
        # Create optimizer
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        # parameters.setRelinearizeSkip(10)
        # parameters.setFactorization("QR")
        parameters.setFactorization("CHOLESKY") # Numerically more stable than QR
        self.isam = gtsam.ISAM2(parameters)

        rospy.loginfo("GTSAM initialized")
    
    def odom_cb(self, msg):
        """Process odometry data"""
        with self.lock:
            self.latest_odom = msg
            
            # Extract pose from odometry
            if self.current_pose_id > 5:
                # After 5 keyframes, add noise
                # position_x = msg.pose.pose.position.x + np.random.normal(0, np.sqrt(self.R_odom[0,0]))
                # position_y = msg.pose.pose.position.y + np.random.normal(0, np.sqrt(self.R_odom[1,1]))
                position_x = msg.pose.pose.position.x
                position_y = msg.pose.pose.position.y
            else:
                position_x = msg.pose.pose.position.x 
                position_y = msg.pose.pose.position.y

            orientation = msg.pose.pose.orientation
            
            # Convert quaternion to RPY
            roll, pitch, yaw = tf.transformations.euler_from_quaternion(
                [orientation.x, orientation.y, orientation.z, orientation.w])
            
            # if self.current_pose_id > 5:
                # After 5 keyframes, add noise
                # yaw += np.random.normal(0, np.sqrt(self.R_odom[2,2]))
            
            current_pose = gtsam.Pose2(position_x, position_y, yaw)
            
            
            # Check if we should add a new keyframe
            if self.current_pose_id == 0:
                # First keyframe
                self.keyframe_poses.append(current_pose)
                self.keyframe_poses_cov.append(np.diag([0.1, 0.1, 0.01]))
                self.keyframe_ground_truth.append(current_pose)
                self.current_pose_id += 1
            else:
                last_pose = self.keyframe_poses[-1]
                delta_pos = np.sqrt((current_pose.x() - last_pose.x())**2 + 
                                    (current_pose.y() - last_pose.y())**2)
                delta_angle = abs(current_pose.theta() - last_pose.theta())

                if self.reached_goal == True:
                    self.reached_goal = False
                    self.plan_path_to_closest_frontier()    # option 1: plan path to closest frontier
                    # self.plan_path_to_frontier()          # option 2: plan path to informative frontier
                
                if delta_pos > self.keyframe_distance or delta_angle > self.keyframe_angle:
                    # Add a new keyframe
                    self.add_keyframe(current_pose, self.latest_scan)
                    self.attempt_loop_closure()
                    self.optimize_graph()
                    if self.current_pose_id % 20 == 1:
                        self.publish_visualization()

                self.time_elapsed_since_movement = time.time()-self.last_marker_time
                if self.time_elapsed_since_movement > self.recovery_time:
                    self.do_recovery()
                    rospy.loginfo("Recovery behavior initiated")

    def scan_cb(self, msg):
        """Process point cloud data"""
        with self.lock:
            try:
                # Transform point cloud to map frame
                transform = self.tf_buffer.lookup_transform("warty/map",  # Target frame
                                                      msg.header.frame_id,  # Source frame
                                                      msg.header.stamp,
                                                      rospy.Duration(1.0))
                
                cloud_map = do_transform_cloud(msg, transform)
                self.latest_scan = cloud_map
                self.latest_transform = transform
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                    tf2_ros.ExtrapolationException) as e:
                rospy.logerr(f"TF Error: {e}")
    
    def add_keyframe(self, pose, scan):
        """Add a keyframe to the graph"""
        
        current_id = self.current_pose_id
        previous_id = current_id - 1
        
        # Add odometry factor
        previous_pose = self.keyframe_poses[-1]
        delta = previous_pose.between(pose)
        self.graph.add(gtsam.BetweenFactorPose2(previous_id, current_id, delta, self.odom_noise))
        
        # Add to initial estimate
        # if not self.initial_estimate.exists(current_id):
        self.initial_estimate.insert(current_id, pose)
        
        # Store keyframe data
        self.keyframe_poses.append(pose)
        self.keyframe_scans.append(self.process_scan(pose, scan))

        # Store keyframe (Ground truth) data
        current_ground_truth = self.x_pos_ground, self.y_pos_ground
        self.keyframe_ground_truth.append(current_ground_truth)
        
        self.current_pose_id += 1
        # rospy.loginfo(f"Added keyframe {self.current_pose_id}")

        # Update the map
        if current_id % self.map_update_frequency == 1:
            self.update_occupancy_map(pose, self.process_scan(pose, scan))
        # rospy.loginfo(f"Updated occupancy map at keyframe {current_id}")

    def process_scan(self, pose, scan_msg):
        """Convert PointCloud2 to a format suitable for scan matching"""
        points = []
        for point in pc2.read_points(scan_msg, field_names=("x", "y", "z"), skip_nans=True):
            
            # with Lidar noise
            # noisy_x = point[0] + np.random.normal(0, np.sqrt(self.lidar_range_noise))
            # noisy_y = point[1] + np.random.normal(0, np.sqrt(self.lidar_range_noise))
            # without Lidar noise
            noisy_x = point[0]
            noisy_y = point[1]
            z = point[2]

            # Filter points (based on height, range)
            if abs(z - 0.15) < 0.01 and np.sqrt((noisy_x - pose.x())**2 + (noisy_y - pose.y())**2) < self.max_range:
                points.append((point[0], point[1]))
        
        return np.array(points)

    def optimize_graph(self):
        """Optimize the pose graph"""
        if self.graph.size() < 1:
            return
            
        try:
            # Store previous poses to detect significant changes
            previous_poses = {}
            for i in range(self.current_pose_id):
                if self.current_estimate.exists(i):
                    previous_poses[i] = self.current_estimate.atPose2(i)
            
            # Optimize using ISAM2
            self.isam.update(self.graph, self.initial_estimate)
            self.current_estimate = self.isam.calculateEstimate()
            
            # Clear the factors and values that were just added
            # self.graph.resize(0)
            # self.initial_estimate.clear()

            # Instead, just clear the initial estimate (since these values are now incorporated)
            self.initial_estimate.clear()
            
            # Get the latest optimized pose
            latest_optimized_pose = None
            if self.current_pose_id > 0 and self.current_estimate.exists(self.current_pose_id - 1):
                latest_optimized_pose = self.current_estimate.atPose2(self.current_pose_id - 1)
                # Publish the optimized pose
                self.publish_pose(latest_optimized_pose)
            
            # Update keyframe poses with optimized values
            self.keyframe_poses = []
            for i in range(self.current_pose_id):
                if self.current_estimate.exists(i):
                    self.keyframe_poses.append(self.current_estimate.atPose2(i))

            # Now we can safely compute covariances
            self.keyframe_poses_cov = []
            marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(), self.current_estimate)
            for i in range(self.current_pose_id):
                if self.current_estimate.exists(i):
                    try:
                        cov_matrix = marginals.marginalCovariance(i)
                        self.keyframe_poses_cov.append(cov_matrix)
                    except Exception as e:
                        rospy.logwarn(f"Could not compute covariance for pose {i}: {e}")
                        self.keyframe_poses_cov.append(np.eye(3))  # Default identity covariance

            # Check if significant position corrections occurred
            significant_change = False
            max_change = 0.0
            max_change_id = -1
            for i in range(self.current_pose_id):
                if self.current_estimate.exists(i) and i in previous_poses:
                    prev = previous_poses[i]
                    curr = self.current_estimate.atPose2(i)
                    
                    # Calculate position change
                    dx = prev.x() - curr.x()
                    dy = prev.y() - curr.y()
                    change = np.sqrt(dx*dx + dy*dy)
                    
                    if change > max_change:
                        max_change = change
                        max_change_id = i
                    
                    dtheta = abs(prev.theta() - curr.theta())
                    
                    # If any pose changed by more than threshold, mark as significant
                    if change > 0.1 or dtheta > 0.05:
                        significant_change = True
                        break
            
            # If significant change detected from loop closure, update map
            # rospy.loginfo(f"Significant change: {significant_change} keyframes: {len(self.keyframe_poses)}")
            if significant_change and len(self.keyframe_poses) > 10:
                rospy.loginfo(f"Significant pose change detected: max {max_change:.3f}m at frame {max_change_id}")
                
                # self.incremental_map_update_after_loop_closure(previous_poses)
                
                # Alternatively, for a complete rebuild (higher quality but slower):
                self.update_map_after_loop_closure()

                # Visualize loop closure events
                for i in range(self.current_pose_id):
                    for j in range(i):
                        if self.current_estimate.exists(i) and self.current_estimate.exists(j):
                            # Only visualize distant keyframes that are now close (potential loop closures)
                            if abs(i - j) > 20:  # Far apart in sequence
                                pose_i = self.current_estimate.atPose2(i)
                                pose_j = self.current_estimate.atPose2(j)
                                dx = pose_i.x() - pose_j.x()
                                dy = pose_i.y() - pose_j.y()
                                distance = np.sqrt(dx*dx + dy*dy)
                                
                                # If they're physically close despite being sequence-distant
                                if distance < 1.0:
                                    self.visualize_loop_closure(i, j, [dx, dy, pose_i.theta() - pose_j.theta()])
                
                # Update visualization after loop closure
                self.publish_visualization()
                
                # Force a scan update with the latest pose to fix current robot position on map
                if latest_optimized_pose is not None and self.latest_scan is not None:
                    # Convert the point cloud to numpy array for processing
                    points_list = []
                    for point in pc2.read_points(self.latest_scan, field_names=("x", "y", "z"), skip_nans=True):
                        if abs(point[2] - 0.15) < 0.01:  # Filter by height
                            points_list.append((point[0], point[1]))
                    
                    if points_list:
                        scan_array = np.array(points_list)
                        self.update_occupancy_map(latest_optimized_pose, scan_array)
            
        except Exception as e:
            rospy.logerr(f"Error in graph optimization: {e}")

    ''' Loop closure functions '''
    def attempt_loop_closure(self):
        """Try to detect loop closures using scan matching"""
        if len(self.keyframe_scans) < 10:
            return False
        
        latest_scan = self.keyframe_scans[-1]
        latest_id = self.current_pose_id - 1
        latest_pose = self.keyframe_poses[-1]
        
        # Only check older keyframes (skip the most recent ones)
        # 1. Testing with 20, then increase it if long-term loops are missing.
        # 2. Use odometry drift as a guide—if drift is large, increase the search range.
        # 3. Log loop closures to analyze if missed ones happen beyond 20 frames.
        candidates = list(range(max(0, latest_id - 20)))
        
        for candidate_id in candidates:
            # Skip very recent frames
            if latest_id - candidate_id < 10:
                continue
                
            candidate_scan = self.keyframe_scans[candidate_id]
            candidate_pose = self.keyframe_poses[candidate_id]
            # Compute relative pose for initial guess
            init_pose = candidate_pose.between(latest_pose)
            
            # Perform scan matching (simple ICP could be used here)
            success, transform = self.match_scans(init_pose, latest_scan, candidate_scan)
            
            if success:
                # Check match quality more rigorously
                distance = np.sqrt(transform[0]**2 + transform[1]**2)
                angle_diff = abs(transform[2])
                
                # if distance > 10.0 or angle_diff > np.pi/2:
                if distance > 2.0 or angle_diff > np.pi/6:
                    # rospy.logwarn(f"Loop closure rejected: too large transform dist={distance:.2f}m angle={np.rad2deg(angle_diff):.1f}°")
                    continue

                # Secondary verification: Consistency check
                # Apply transformation to candidate_scan and check overlap with latest_scan
                # c, s = np.cos(transform[2]), np.sin(transform[2])
                # R = np.array([[c, -s], [s, c]])
                # transformed_points = (R @ candidate_scan.T).T + np.array([transform[0], transform[1]])
                
                # # Calculate overlap score
                # overlap_score = self.calculate_scan_overlap(transformed_points, latest_scan)
                # if overlap_score < 0.7:
                #     # rospy.logwarn(f"Loop closure rejected: low overlap score {overlap_score:.2f}")
                #     continue
                    
                rospy.loginfo(f"Loop closure detected between frame {latest_id} and {candidate_id}, "
                            f"transform=({transform[0]:.2f}, {transform[1]:.2f}, {np.rad2deg(transform[2]):.1f}°)")
                
                # Add loop closure constraint
                rel_pose = gtsam.Pose2(transform[0], transform[1], transform[2])
                self.graph.add(gtsam.BetweenFactorPose2(latest_id, candidate_id, rel_pose, self.loop_noise))
                
                # Visualize the loop closure
                self.visualize_loop_closure(latest_id, candidate_id, transform)
                
                return True
                
        return False
    
    def match_scans(self, pose, scan1, scan2):
        """
        Match two scans using NumPy-based ICP algorithm for loop closure detection.
        
        Args:
            pose: Initial pose estimate (GTSAM Pose2 object)
            scan1: First point cloud as Nx2 numpy array (target)
            scan2: Second point cloud as Nx2 numpy array (source to be aligned)
            
        Returns:
            success: Boolean indicating whether matching was successful
            transform: [dx, dy, dtheta] transformation to align scan2 to scan1
        """
        # Early termination if scans are too small
        if len(scan1) < 10 or len(scan2) < 10:
            rospy.logwarn("Not enough points for scan matching")
            return False, None
            
        try:
            # Parameters
            max_iterations = 30
            tolerance = 0.001
            outlier_threshold = 1.0
            # min_match_quality = 0.6  # Minimum quality of match (0-1)
            min_match_quality = 0.8  # Minimum quality of match (0-1)
            
            # Initialize transformation with the given pose
            current_rotation = pose.theta()
            current_translation = np.array([pose.x(), pose.y()])
            
            # Create copies to avoid modifying original data
            source_points = np.copy(scan2)
            target_points = np.copy(scan1)
            
            # Downsample scans for efficiency using stride-based approach
            if len(source_points) > 1000:
                stride = max(1, len(source_points) // 1000)
                source_points = source_points[::stride]
            
            if len(target_points) > 1000:
                stride = max(1, len(target_points) // 1000)
                target_points = target_points[::stride]
                
            # Pre-compute squared distances for efficiency
            def squared_dist(x, y):
                return (x[:, np.newaxis] - y[np.newaxis, :])**2
                
            # Track convergence and quality
            prev_error = float('inf')
            
            # ICP main loop
            for iteration in range(max_iterations):
                # Create rotation matrix from current angle
                c, s = np.cos(current_rotation), np.sin(current_rotation)
                R = np.array([[c, -s], [s, c]])
                
                # Apply current transformation to source points
                transformed_source = (R @ source_points.T).T + current_translation
                
                # Find closest points using vectorized operations
                # Calculate squared distances for each source point to all target points
                dx = squared_dist(transformed_source[:, 0], target_points[:, 0])
                dy = squared_dist(transformed_source[:, 1], target_points[:, 1])
                distances = np.sqrt(dx + dy)
                
                # Get index of closest target point for each source point
                min_dist_idx = np.argmin(distances, axis=1)
                min_distances = np.min(distances, axis=1)
                
                # Filter out outliers
                valid_idx = min_distances < outlier_threshold
                if np.sum(valid_idx) < 5:
                    # rospy.logwarn(f"Too few correspondences: {np.sum(valid_idx)}")
                    return False, None
                    
                src_valid = transformed_source[valid_idx]
                tgt_valid = target_points[min_dist_idx[valid_idx]]
                
                # Calculate mean correspondance distance for convergence check
                mean_error = np.mean(min_distances[valid_idx])
                if abs(prev_error - mean_error) < tolerance:
                    break
                prev_error = mean_error
                
                # Calculate centroids
                src_centroid = np.mean(src_valid, axis=0)
                tgt_centroid = np.mean(tgt_valid, axis=0)
                
                # Center the point clouds
                src_centered = src_valid - src_centroid
                tgt_centered = tgt_valid - tgt_centroid
                
                # Calculate covariance matrix efficiently
                H = src_centered.T @ tgt_centered
                
                # SVD decomposition
                try:
                    U, _, Vt = np.linalg.svd(H)
                    
                    # Calculate rotation matrix
                    R_opt = Vt.T @ U.T
                    
                    # Handle reflection case
                    if np.linalg.det(R_opt) < 0:
                        Vt[-1, :] *= -1
                        R_opt = Vt.T @ U.T
                    
                    # Calculate rotation angle
                    theta_opt = np.arctan2(R_opt[1, 0], R_opt[0, 0])
                    
                    # Calculate translation
                    t_opt = tgt_centroid - R_opt @ src_centroid
                    
                    # Update transformation
                    current_rotation += theta_opt
                    # Normalize the angle
                    current_rotation = np.arctan2(np.sin(current_rotation), np.cos(current_rotation))
                    current_translation = t_opt + R_opt @ current_translation
                    
                except np.linalg.LinAlgError:
                    rospy.logwarn("SVD computation failed")
                    continue
            
            # Calculate final match quality
            # Count inliers and calculate proportion
            inlier_count = np.sum(valid_idx)
            match_quality = inlier_count / float(len(source_points))
            
            # rospy.loginfo(f"ICP iteration={iteration+1}, error={mean_error:.4f}, quality={match_quality:.2f}")
            
            # Return transformation only if match quality is sufficient
            if match_quality > min_match_quality and mean_error < outlier_threshold * 0.5:
                return True, [current_translation[0], current_translation[1], current_rotation]
            else:
                return False, None
                
        except Exception as e:
            rospy.logerr(f"Error in ICP scan matching: {e}")
            return False, None

    def visualize_loop_closure(self, id1, id2, transform):
        """Visualize a loop closure between two frames"""
        if not hasattr(self, 'loop_pub'):
            self.loop_pub = rospy.Publisher('/graph_slam/loop_closures', Marker, queue_size=10)
        
        # Get poses of frames involved in loop closure
        if self.current_estimate.exists(id1) and self.current_estimate.exists(id2):
            pose1 = self.current_estimate.atPose2(id1)
            pose2 = self.current_estimate.atPose2(id2)
            
            # Create line marker connecting the two poses
            marker = Marker()
            marker.header.frame_id = "warty/map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "loop_closures"
            marker.id = id1 * 1000 + id2  # Unique ID
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.1  # Line width
            marker.color.r = 0.0  # Red color
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.7
            marker.lifetime = rospy.Duration(0)  # Persist forever
            
            # Add the two poses as points in the line
            p1 = Point()
            p1.x, p1.y = pose1.x(), pose1.y()
            p1.z = 0.1  # Slightly above ground
            
            p2 = Point()
            p2.x, p2.y = pose2.x(), pose2.y()
            p2.z = 0.1
            
            marker.points = [p1, p2]
            
            # Publish the marker
            self.loop_pub.publish(marker)

    def update_map_after_loop_closure(self):
        """Regenerate the occupancy grid map after loop closure"""
        if not hasattr(self, 'keyframe_scans') or len(self.keyframe_scans) == 0:
            rospy.logwarn("No keyframes available for map update")
            return

        rospy.loginfo("Regenerating map after loop closure")
        
        # Reset the log odds map to initial state
        self.log_odds_map = np.zeros((self.height, self.width))
        self.global_map = np.full((self.height, self.width), -1, dtype=np.int8)
        
        # Process all keyframes
        for i, (pose, scan) in enumerate(zip(self.keyframe_poses, self.keyframe_scans)):
            if i % 5 == 0:  # Process every 5th keyframe to save computation
                rospy.loginfo(f"Processing keyframe {i}/{len(self.keyframe_poses)}")
                
                # Process each point in the point cloud
                for point in scan:
                    x, y = point[:2]
                    grid_x = int((x - self.origin.position.x) / self.resolution)
                    grid_y = int((y - self.origin.position.y) / self.resolution)
                    
                    if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                        # Update log odds for occupied cell
                        self.log_odds_map[grid_y, grid_x] += self.log_odds_occupied*self.reupdate_factor
                        # Apply clamping
                        self.log_odds_map[grid_y, grid_x] = min(self.log_odds_map[grid_y, grid_x], self.log_odds_max)
                
                # Raycast for free space
                sensor_x = int((pose.x() - self.origin.position.x) / self.resolution)
                sensor_y = int((pose.y() - self.origin.position.y) / self.resolution)
                
                if 0 <= sensor_x < self.width and 0 <= sensor_y < self.height:
                    for point in scan:
                        x, y = point[:2]
                        target_x = int((x - self.origin.position.x) / self.resolution)
                        target_y = int((y - self.origin.position.y) / self.resolution)
                        
                        if 0 <= target_x < self.width and 0 <= target_y < self.height:
                            # Use Bresenham's line algorithm for raycasting
                            rr, cc = line(sensor_y, sensor_x, target_y, target_x)
                            for j in range(len(rr) - 1):  # All but the last point
                                if 0 <= rr[j] < self.height and 0 <= cc[j] < self.width:
                                    # Update log odds for free cell
                                    self.log_odds_map[rr[j], cc[j]] += self.log_odds_free*self.reupdate_factor
                                    # Apply clamping
                                    self.log_odds_map[rr[j], cc[j]] = max(self.log_odds_map[rr[j], cc[j]], self.log_odds_min)
        
        # Convert log odds to probability
        unknown_mask = np.isclose(self.log_odds_map, 0.0)  # Prior value of 0.0
        probability_map = np.full_like(self.global_map, -1, dtype=np.int8)
        known_mask = ~unknown_mask
        
        if np.any(known_mask):
            probability_values = 100 * (1 - 1 / (1 + np.exp(self.log_odds_map[known_mask])))
            probability_map[known_mask] = probability_values.astype(np.int8)
        
        # Update global map
        self.global_map = probability_map
        
        # Publish updated map
        grid_msg = OccupancyGrid()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "warty/map"
        grid_msg.header = header

        grid_msg.info = self.map_meta_data
        grid_msg.info.height = self.global_map.shape[0]
        grid_msg.info.width = self.global_map.shape[1]
        grid_msg.data = self.global_map.flatten().tolist()
        
        self.map_publisher.publish(grid_msg)
        rospy.loginfo("Map regenerated after loop closure")

    def incremental_map_update_after_loop_closure(self, previous_poses):
        """Update only cells affected by significant pose changes"""
        rospy.loginfo("Incrementally updating map after loop closure")
        
        # Find poses with significant changes
        changed_keyframes = []
        for i, old_pose in previous_poses.items():
            if i < len(self.keyframe_poses):
                new_pose = self.keyframe_poses[i]
                dx = old_pose.x() - new_pose.x()
                dy = old_pose.y() - new_pose.y()
                dtheta = abs(old_pose.theta() - new_pose.theta())
                
                # If pose changed significantly
                if np.sqrt(dx*dx + dy*dy) > 0.05 or dtheta > 0.02:
                    changed_keyframes.append(i)
        
        # Skip if no significant changes
        if not changed_keyframes:
            rospy.loginfo("No significant pose changes to update")
            return
        
        rospy.loginfo(f"Updating map for {len(changed_keyframes)} changed keyframes")
        
        # Define log odds parameters
        log_odds_occupied = 1.0
        log_odds_free = -0.7
        log_odds_max = 5.0
        log_odds_min = -5.0
        
        # Create temporary log odds map to track updates
        temp_log_odds_map = np.zeros((self.height, self.width))
        
        # Process only the changed keyframes
        for i in changed_keyframes:
            if i < len(self.keyframe_scans):
                pose = self.keyframe_poses[i]
                scan = self.keyframe_scans[i]
                
                # First, remove old scan data from the log odds map
                old_pose = previous_poses[i]
                
                # Process each point in the point cloud with the new pose
                for point in scan:
                    x, y = point[:2]
                    
                    # Convert to grid coordinates
                    grid_x = int((x - self.origin.position.x) / self.resolution)
                    grid_y = int((y - self.origin.position.y) / self.resolution)
                    
                    if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                        # Mark cell as occupied in temporary map
                        temp_log_odds_map[grid_y, grid_x] += log_odds_occupied
                        # Apply clamping
                        temp_log_odds_map[grid_y, grid_x] = min(temp_log_odds_map[grid_y, grid_x], log_odds_max)
                
                # Raycast for free space
                sensor_x = int((pose.x() - self.origin.position.x) / self.resolution)
                sensor_y = int((pose.y() - self.origin.position.y) / self.resolution)
                
                if 0 <= sensor_x < self.width and 0 <= sensor_y < self.height:
                    for point in scan:
                        x, y = point[:2]
                        target_x = int((x - self.origin.position.x) / self.resolution)
                        target_y = int((y - self.origin.position.y) / self.resolution)
                        
                        if 0 <= target_x < self.width and 0 <= target_y < self.height:
                            # Use Bresenham's line algorithm for raycasting
                            rr, cc = line(sensor_y, sensor_x, target_y, target_x)
                            for j in range(len(rr) - 1):  # All but the last point
                                if 0 <= rr[j] < self.height and 0 <= cc[j] < self.width:
                                    # Update log odds for free cell
                                    temp_log_odds_map[rr[j], cc[j]] += log_odds_free
                                    # Apply clamping
                                    temp_log_odds_map[rr[j], cc[j]] = max(temp_log_odds_map[rr[j], cc[j]], log_odds_min)
        
        # Apply changes to main log odds map
        # Find cells that have been updated in temp map (non-zero values)
        updated_cells = temp_log_odds_map != 0
        
        if np.any(updated_cells):
            # Replace only the updated cells in the main log odds map
            self.log_odds_map[updated_cells] = temp_log_odds_map[updated_cells]
        
        # Convert updated log odds to probability
        unknown_mask = np.isclose(self.log_odds_map, 0.0)  # Prior value of 0.0
        probability_map = np.full_like(self.global_map, -1, dtype=np.int8)
        known_mask = ~unknown_mask
        
        if np.any(known_mask):
            probability_values = 100 * (1 - 1 / (1 + np.exp(self.log_odds_map[known_mask])))
            probability_map[known_mask] = probability_values.astype(np.int8)
        
        # Update global map
        self.global_map = probability_map
        
        # Publish updated map
        grid_msg = OccupancyGrid()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "warty/map"
        grid_msg.header = header

        grid_msg.info = self.map_meta_data
        grid_msg.info.height = self.global_map.shape[0]
        grid_msg.info.width = self.global_map.shape[1]
        grid_msg.data = self.global_map.flatten().tolist()
        
        self.map_publisher.publish(grid_msg)
        rospy.loginfo("Map incrementally updated after loop closure")

    ''' Visualization functions '''
    def publish_pose(self, pose):
        """Publish the latest optimized pose"""
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "warty/map"
        pose_stamped.header.stamp = rospy.Time.now()
        
        pose_stamped.pose.position.x = pose.x()
        pose_stamped.pose.position.y = pose.y()
        pose_stamped.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        quat = tf.transformations.quaternion_from_euler(0, 0, pose.theta())
        pose_stamped.pose.orientation.x = quat[0]
        pose_stamped.pose.orientation.y = quat[1]
        pose_stamped.pose.orientation.z = quat[2]
        pose_stamped.pose.orientation.w = quat[3]
        
        self.pose_publisher.publish(pose_stamped)

    def publish_visualization(self):
        """Publish visualization of the graph and trajectory"""
        marker_array = MarkerArray()
        
        # Trajectory line
        path_marker = Marker()
        path_marker.header.frame_id = "warty/map"
        path_marker.header.stamp = rospy.Time.now()
        path_marker.ns = "trajectory"
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.05  # Line width
        path_marker.color.r = 1.0
        path_marker.color.g = 0.0
        path_marker.color.b = 0.0
        path_marker.color.a = 0.1
        path_marker.pose.orientation.w = 1.0
        
        # Node markers
        node_marker = Marker()
        node_marker.header.frame_id = "warty/map"
        node_marker.header.stamp = rospy.Time.now()
        node_marker.ns = "nodes"
        node_marker.id = 1
        node_marker.type = Marker.POINTS
        node_marker.action = Marker.ADD
        node_marker.scale.x = 0.2  # Point size
        node_marker.scale.y = 0.2
        node_marker.color.r = 1.0
        node_marker.color.g = 0.0
        node_marker.color.b = 0.0
        node_marker.color.a = 1.0
        
        # Loop closure markers
        edges_marker = Marker()
        edges_marker.header.frame_id = "warty/map"
        edges_marker.header.stamp = rospy.Time.now()
        edges_marker.ns = "loop_closures"
        edges_marker.id = 2
        edges_marker.type = Marker.LINE_LIST
        edges_marker.action = Marker.ADD
        edges_marker.scale.x = 0.05  # Line width
        edges_marker.color.r = 0.0
        edges_marker.color.g = 0.0
        edges_marker.color.b = 1.0
        edges_marker.color.a = 1.0
        
        # Add points to markers
        for i in range(self.current_pose_id):
            if self.current_estimate.exists(i):
                pose = self.current_estimate.atPose2(i)
                
                # Add to path
                point = Point()
                point.x = pose.x()
                point.y = pose.y()
                point.z = 0.0
                path_marker.points.append(point)
                
                # Add to nodes
                node_marker.points.append(point)
        
        marker_array.markers.append(path_marker)
        marker_array.markers.append(node_marker)
        marker_array.markers.append(edges_marker)
        
        self.viz_publisher.publish(marker_array)
        self.path_publisher.publish(path_marker)
        # rospy.loginfo("Published visualization markers")

    def marker_cb(self,msg):
        ''' Extract frontier locations and save it as a list of frontiers '''
        self.last_marker_time=time.time()
        frontiers=msg.points
        x_array=[]
        y_array=[]
        for point in frontiers:
            x_array.append(point.x)
            y_array.append(point.y) 
        
        print('000000000000000000000000000000000000000000000000')
        # print('frontier x','frontier y',x_array,y_array)
        
        # Convert list to numpy array
        self.frontier_x=np.array(x_array)
        self.frontier_y=np.array(y_array)

    def ground_truth_cb(self, msg):
        ''' ground truth robot position '''

        self.x_pos_ground = msg.pose.position.x - self.offset_x
        self.y_pos_ground = msg.pose.position.y - self.offset_y

        self.z_pos_ground = msg.pose.position.z

        self.orient_w_ground = msg.pose.orientation.w
        self.orient_x_ground = msg.pose.orientation.x
        self.orient_y_ground = msg.pose.orientation.y
        self.orient_z_ground = msg.pose.orientation.z

        # From "quaternion" to "euler"
        t3 = +2.0 * (self.orient_w_ground * self.orient_z_ground + self.orient_x_ground * self.orient_y_ground)
        t4 = +1.0 - 2.0 * (self.orient_y_ground * self.orient_y_ground + self.orient_z_ground * self.orient_z_ground)
        self.yaw_z_ground = math.atan2(t3, t4)

    ''' Mapping functions '''
    def costmap_cb(self, msg):
        """Process the costmap and update the occupancy grid map"""

        if not self.costmap_initialized:
            if self.sim_env == 1:
                self.height = msg.info.height
                self.width = msg.info.width
            elif self.sim_env == 2:
                self.height = 847       # 32 rooms
                self.width = 625
            elif self.sim_env == 3:
                self.height = 1454         # uc1
                self.width = 1101

            self.resolution = msg.info.resolution
            self.origin = msg.info.origin

            if self.sim_env == 1 or self.sim_env == 2:
                self.origin.position.x = self.origin.position.x  + self.offset_x
                self.origin.position.y = self.origin.position.y  + self.offset_y
            elif self.sim_env == 3:
                self.origin.position.x = self.origin.position.x + self.offset_x + 50.0  # uc1
                self.origin.position.y = -227.06
            
            self.map_meta_data = msg.info

            rospy.loginfo("inside height: %d, width: %d, env: %d", self.height, self.width, self.sim_env)
            rospy.loginfo("origin_x: %f, origin_y: %f", self.origin.position.x, self.origin.position.y)

            # Initialize the global occupancy grid
            self.global_map = np.full((self.height, self.width), -1, dtype=np.int8)

            self.costmap_initialized = True        
        
    def update_occupancy_map(self, pose, point_cloud):
        """Update and publish the occupancy grid map"""
        if point_cloud.size == 0:
            return
        
        # Get the latest optimized pose if it exists, otherwise use the provided pose
        current_pose = pose
        if hasattr(self, 'current_estimate') and self.current_pose_id > 0:
            latest_pose_id = self.current_pose_id - 1
            if self.current_estimate.exists(latest_pose_id):
                optimized_pose = self.current_estimate.atPose2(latest_pose_id)
                
                # Check if there's a significant difference between the poses
                dx = optimized_pose.x() - pose.x()
                dy = optimized_pose.y() - pose.y()
                dtheta = abs(optimized_pose.theta() - pose.theta())
                
                if np.sqrt(dx*dx + dy*dy) > 0.1 or dtheta > 0.05:
                    rospy.loginfo(f"Using optimized pose for map update. Diff: {np.sqrt(dx*dx + dy*dy):.3f}m, {np.rad2deg(dtheta):.1f}°")
                    current_pose = optimized_pose
                    
                    # Transform point cloud to account for pose correction
                    c1, s1 = np.cos(pose.theta()), np.sin(pose.theta())
                    c2, s2 = np.cos(optimized_pose.theta()), np.sin(optimized_pose.theta())
                    
                    # Remove the original pose transform
                    point_cloud_local = point_cloud.copy()
                    point_cloud_local[:, 0] -= pose.x() 
                    point_cloud_local[:, 1] -= pose.y()
                    
                    # Rotate back to the robot frame
                    x_rot = point_cloud_local[:, 0] * c1 + point_cloud_local[:, 1] * s1
                    y_rot = -point_cloud_local[:, 0] * s1 + point_cloud_local[:, 1] * c1
                    
                    # Apply the optimized pose transform
                    x_new = x_rot * c2 - y_rot * s2 + optimized_pose.x()
                    y_new = x_rot * s2 + y_rot * c2 + optimized_pose.y()
                    
                    point_cloud = np.column_stack((x_new, y_new))
        
        # Convert to occupancy grid using log odds update
        occupancy_grid = self.point_cloud_to_occupancy_map_log_odd(current_pose, point_cloud, self.resolution)

        # Publish the grid
        if occupancy_grid is not None:
            self.map_publisher.publish(occupancy_grid)

    def point_cloud_to_occupancy_map(self, pose, point_cloud, resolution):
        """Convert a PointCloud2 message to an occupancy grid map"""
        
        # Determine grid bounds from points
        points_array = point_cloud
        
        # Mark cells with points as occupied (100)
        for point in points_array:
            x, y = point[:2]
            grid_x = int((x - self.origin.position.x) / resolution)
            grid_y = int((y - self.origin.position.y) / resolution)
                
            if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                self.global_map[grid_y, grid_x] = 100  # Occupied
        
        # Perform raycasting to mark free space (0)
        current_pose = pose
        # rospy.loginfo(f"Current pose: {current_pose.x()}, {current_pose.y()}")
        if current_pose is not None:
            sensor_x = int((current_pose.x() - self.origin.position.x) / resolution)
            sensor_y = int((current_pose.y() - self.origin.position.y) / resolution)
            # rospy.loginfo(f"Sensor position: {sensor_x}, {sensor_y}")
            
            if 0 <= sensor_x < self.width and 0 <= sensor_y < self.height:
                for point in points_array:
                    x, y = point[:2]
                    target_x = int((x - self.origin.position.x) / resolution)
                    target_y = int((y - self.origin.position.y) / resolution)
                    
                    if 0 <= target_x < self.width and 0 <= target_y < self.height:
                        # Use Bresenham's line algorithm to raycast
                        rr, cc = line(sensor_y, sensor_x, target_y, target_x)
                        for i in range(len(rr) - 1):  # All but the last point (occupied)
                            if 0 <= rr[i] < self.height and 0 <= cc[i] < self.width:
                                if self.global_map[rr[i], cc[i]] == -1:
                                    self.global_map[rr[i], cc[i]] = 0  # Free
        
        # Create ROS OccupancyGrid message
        grid_msg = OccupancyGrid()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "warty/map"
        grid_msg.header = header

        grid_msg.info = self.map_meta_data
        grid_msg.info.height = self.global_map.shape[0]
        grid_msg.info.width = self.global_map.shape[1]
        grid_msg.data = self.global_map.flatten().tolist()
        
        return grid_msg

    def point_cloud_to_occupancy_map_log_odd(self, pose, point_cloud, resolution):
        """Convert a PointCloud2 message to an occupancy grid map using log odds updates with parallel processing"""
            
        # Initialize log odds map if needed
        if not hasattr(self, 'log_odds_map'):
            self.log_odds_map = np.zeros((self.height, self.width))
            
            # Initialize from existing map if available
            if hasattr(self, 'global_map'):
                occupied_mask = (self.global_map == 100)
                free_mask = (self.global_map == 0)
                unknown_mask = (self.global_map == -1)
                
                self.log_odds_map[occupied_mask] = self.log_odds_occupied
                self.log_odds_map[free_mask] = self.log_odds_free
                self.log_odds_map[unknown_mask] = self.log_odds_prior
        
        # Extract sensor position
        sensor_x = int((pose.x() - self.origin.position.x) / resolution)
        sensor_y = int((pose.y() - self.origin.position.y) / resolution)
        
        # Points outside the valid map area
        if not (0 <= sensor_x < self.width and 0 <= sensor_y < self.height):
            rospy.logwarn(f"Sensor position {sensor_x}, {sensor_y} outside map bounds")
            return None
           
        # Define worker function for parallel processing
        def process_point(point):
            x, y = point[:2]
            grid_x = int((x - self.origin.position.x) / resolution)
            grid_y = int((y - self.origin.position.y) / resolution)
            
            if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                # Mark as occupied
                if self.log_odds_map[grid_y, grid_x] > self.log_odds_min:
                    self.log_odds_map[grid_y, grid_x] += self.log_odds_occupied
                    self.log_odds_map[grid_y, grid_x] = min(self.log_odds_map[grid_y, grid_x], self.log_odds_max)
                
                # Raycasting
                self.log_odds_re_update = False
                if 0 <= sensor_x < self.width and 0 <= sensor_y < self.height:
                    rr, cc = line(sensor_y, sensor_x, grid_y, grid_x)
                    for i in range(len(rr) - 1):  # All but the last point
                        if 0 <= rr[i] < self.height and 0 <= cc[i] < self.width:
                            if self.log_odds_map[rr[i], cc[i]] >= self.log_odds_threshold:
                                break
                            else:
                                self.log_odds_map[rr[i], cc[i]] += self.log_odds_free
                                self.log_odds_map[rr[i], cc[i]] = max(self.log_odds_map[rr[i], cc[i]], self.log_odds_min)
        
        # Parallel processing with thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
            futures = [executor.submit(process_point, point) for point in point_cloud]
            # Optionally, wait for all tasks to complete
            concurrent.futures.wait(futures)
               
        # Convert log odds to probability
        unknown_mask = np.isclose(self.log_odds_map, self.log_odds_prior)
        probability_map = np.full_like(self.global_map, -1, dtype=np.int8)
        known_mask = ~unknown_mask
        
        if np.any(known_mask):
            probability_values = 100 * (1 - 1 / (1 + np.exp(self.log_odds_map[known_mask])))
            probability_map[known_mask] = probability_values.astype(np.int8)


        # Calculate the total map entropy
        map_entropy_comp = np.copy(probability_map)
        map_entropy_comp[map_entropy_comp == -1] = 50
        map_entropy_comp = map_entropy_comp / 100
        
        self.total_map_entorpy = self.shannon_entropy_calculator(map_entropy_comp)
        # rospy.loginfo(f"probability_map: {probability_map} and size of probability_map: {probability_map.size}")
        rospy.loginfo(f"Total map entropy: {self.total_map_entorpy}")
        
        if self.current_pose_id % 10 == 0:

            # Update the global map
            self.global_map = probability_map
            
            # Create ROS OccupancyGrid message
            grid_msg = OccupancyGrid()
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "warty/map"
            grid_msg.header = header
            grid_msg.info = self.map_meta_data
            grid_msg.info.height = self.global_map.shape[0]
            grid_msg.info.width = self.global_map.shape[1]
            grid_msg.data = self.global_map.flatten().tolist()
        else:
            grid_msg = None
        
        return grid_msg

    def point_cloud_to_occupancy_map_log_odd_seq(self, pose, point_cloud, resolution):
        """Convert a PointCloud2 message to an occupancy grid map using log odds updates"""
        
        # Log odds parameters
        log_odds_occupied = 1.0  # Slightly reduced from 1.0
        log_odds_free = -0.7     # Slightly reduced from -0.7
        log_odds_prior = 0.0
        log_odds_max = 5.0
        log_odds_min = -5.0
        
        # Initialize log odds map if needed
        if not hasattr(self, 'log_odds_map'):
            self.log_odds_map = np.zeros((self.height, self.width))
            
            # Initialize from existing map if available
            if hasattr(self, 'global_map'):
                occupied_mask = (self.global_map == 100)
                free_mask = (self.global_map == 0)
                unknown_mask = (self.global_map == -1)
                
                self.log_odds_map[occupied_mask] = log_odds_occupied
                self.log_odds_map[free_mask] = log_odds_free
                self.log_odds_map[unknown_mask] = log_odds_prior
        
        # Extract sensor position
        sensor_x = int((pose.x() - self.origin.position.x) / resolution)
        sensor_y = int((pose.y() - self.origin.position.y) / resolution)
        
        # Points outside the valid map area
        if not (0 <= sensor_x < self.width and 0 <= sensor_y < self.height):
            rospy.logwarn(f"Sensor position {sensor_x}, {sensor_y} outside map bounds")
            return None
        
        # Store the original log odds values for cells that will be updated
        # This allows reverting changes if the update causes inconsistencies
        cells_to_update = set()
        
        # Mark cells with points as occupied
        for point in point_cloud:
            x, y = point[:2]
            grid_x = int((x - self.origin.position.x) / resolution)
            grid_y = int((y - self.origin.position.y) / resolution)
            
            if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                cells_to_update.add((grid_y, grid_x))
                self.log_odds_map[grid_y, grid_x] += log_odds_occupied
                self.log_odds_map[grid_y, grid_x] = min(self.log_odds_map[grid_y, grid_x], log_odds_max)
        
        # Raycast to mark free space
        for point in point_cloud:
            x, y = point[:2]
            target_x = int((x - self.origin.position.x) / resolution)
            target_y = int((y - self.origin.position.y) / resolution)
            
            if 0 <= target_x < self.width and 0 <= target_y < self.height:
                # Use Bresenham's line algorithm
                rr, cc = line(sensor_y, sensor_x, target_y, target_x)
                for i in range(len(rr) - 1):  # All but the last point
                    if 0 <= rr[i] < self.height and 0 <= cc[i] < self.width:
                        cells_to_update.add((rr[i], cc[i]))
                        self.log_odds_map[rr[i], cc[i]] += log_odds_free
                        self.log_odds_map[rr[i], cc[i]] = max(self.log_odds_map[rr[i], cc[i]], log_odds_min)
        
        # Convert log odds to probability
        unknown_mask = np.isclose(self.log_odds_map, log_odds_prior)
        probability_map = np.full_like(self.global_map, -1, dtype=np.int8)
        known_mask = ~unknown_mask
        
        if np.any(known_mask):
            probability_values = 100 * (1 - 1 / (1 + np.exp(self.log_odds_map[known_mask])))
            probability_map[known_mask] = probability_values.astype(np.int8)
        
        # Update the global map
        self.global_map = probability_map
        
        # Create ROS OccupancyGrid message
        grid_msg = OccupancyGrid()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "warty/map"
        grid_msg.header = header
        grid_msg.info = self.map_meta_data
        grid_msg.info.height = self.global_map.shape[0]
        grid_msg.info.width = self.global_map.shape[1]
        grid_msg.data = self.global_map.flatten().tolist()
        
        return grid_msg

    ''' Control related functions'''
    def plan_path_to_closest_frontier(self):
        """Plan a path to the best frontier"""
        if not hasattr(self, 'frontier_x') or len(self.frontier_x) == 0:
            rospy.logwarn("No frontiers available for planning")
            return False
        
        # After constructing the costmap, find the closest frontier
        if self.costmap_initialized:
        
            # Get current position estimate from SLAM
            current_pose = None
            if len(self.keyframe_poses) > 0:
                current_pose = self.keyframe_poses[-1]
                current_x = current_pose.x()
                current_y = current_pose.y()
            else:
                rospy.logwarn("No pose estimate available")
                return False
            
            current_x_grid = int((current_x - self.origin.position.x) / self.resolution)
            current_y_grid = int((current_y - self.origin.position.y) / self.resolution)

            goals = [(self.frontier_x[i], self.frontier_y[i]) for i in range(len(self.frontier_x))]

            cost_map = np.copy(self.global_map)
            cost_map[cost_map == -1] = 49  # Unknown cells are free
            cost_map += 1  # Add 1 to all cells to avoid zero values
            cost_map = np.array(cost_map, dtype=np.float32)

            distance_path = np.inf
            for goal in goals:
                goal_x = int((goal[0] - self.origin.position.x) / self.resolution)
                goal_y = int((goal[1] - self.origin.position.y) / self.resolution)
                path = pyastar2d.astar_path(cost_map, (current_y_grid, current_x_grid), (goal_y, goal_x), allow_diagonal=True)
                distance_path_new = len(path)
                if distance_path_new < distance_path:
                    distance_path = distance_path_new
                    best_goal = goal

            # # Use vectorized operations to compute distances to all frontiers
            # frontiers_x = np.array(self.frontier_x)
            # frontiers_y = np.array(self.frontier_y)

            # # Calculate Euclidean distance using vectorization
            # dx = frontiers_x - current_x
            # dy = frontiers_y - current_y
            # distances = np.sqrt(dx**2 + dy**2)

            # closest_idx = np.argmin(distances)

            # Get frontiers as goal points
            # best_goal = (self.frontier_x[closest_idx], self.frontier_y[closest_idx])
            rospy.loginfo(f"Selected closest frontier: {best_goal}, distance: {distance_path}")
            
            # After finding best action policy, go to the next step
            # Moving the robot based on the best action policy
            if best_goal is not None:
                self.goal_x, self.goal_y = best_goal

                self.client = actionlib.SimpleActionClient('/warty/goto_region', nav_msg.GotoRegionAction)
                rospy.loginfo("Waiting for goto_region action server...")
                wait = self.client.wait_for_server()
                if not wait:
                    rospy.logerr("Action server not available!")
                    rospy.signal_shutdown("Action server not available!")
                    return False
                rospy.loginfo("Connected to action server")
                rospy.loginfo("Starting goals achievements ...")
                self.gotoregion_client()
                return True
            else:
                self.reached_goal = True
                return False
    
    def plan_path_to_frontier(self):
        """Plan a path to the best frontier"""
        if not hasattr(self, 'frontier_x') or len(self.frontier_x) == 0:
            rospy.logwarn("No frontiers available for planning")
            return False
        
        # Get current position estimate from SLAM
        current_pose = None
        if len(self.keyframe_poses) > 0:
            current_pose = self.keyframe_poses[-1]
        else:
            rospy.logwarn("No pose estimate available")
            return False
        
        pred_pos = np.array([current_pose.x(), current_pose.y(), current_pose.theta()])
        marginals = gtsam.Marginals(self.graph, self.keyframe_poses[-1])
        pred_cov = marginals.marginalCovariance(self.current_pose_id)

        rospy.loginfo(f"covariance: {pred_cov}")
    

        # Get frontiers as goal points
        goals = [(self.frontier_x[i], self.frontier_y[i]) for i in range(len(self.frontier_x))]
        rospy.loginfo(f"Planning with {len(goals)} frontier goals")
        
        # best_goal = None
        # if goals:
        #     Max_entropy = -np.inf
        #     # Process goals in parallel for efficiency
        #     with concurrent.futures.ThreadPoolExecutor() as executor:
        #         futures = [executor.submit(self.MPC_compute_res_entropy, goal, pred_pos, pred_pos_index, pred_cov) 
        #                 for goal in goals]
        #         for future in concurrent.futures.as_completed(futures):
        #             res_entropy, goal = future.result()
        #             if res_entropy > Max_entropy:
        #                 Max_entropy = res_entropy
        #                 best_goal = goal
            
        #     rospy.loginfo(f"Selected goal: {best_goal}, with utility: {Max_entropy}")
            
        #     # Set the selected goal
        #     if best_goal is not None:
        #         self.goal_x, self.goal_y = best_goal
        #         return True
        
        return False

    def path_cb(self, msg):
        path = msg.points
        x_array=[]
        y_array=[]
        for point in path:
            x_array.append(point.x)
            y_array.append(point.y) 
        self.path_x_coord=np.array(x_array)
        self.path_y_coord=np.array(y_array)
        # rospy.loginfo(f"Path received with {len(self.path_x_coord)} points---------------------------------")

    def gotoregion_client(self):
        
        ''' Main function'''
        goal_point=Pose()

        vel_msg=Twist()
        vel_msg.linear.x=0.01
        
        # position
        goal_point.position.x= self.goal_x
        goal_point.position.y= self.goal_y
        goal_point.position.z= 0.0
        # Goal orientation
        goal_point.orientation.x= 0.0
        goal_point.orientation.y= 0.0
        goal_point.orientation.z= 0.6
        goal_point.orientation.w= 0.8

        goal_msg = nav_msg.GotoRegionGoal()
        
        # Radius and angle threshold
        goal_rad=self.determine_goal_radius(self.goal_x, self.goal_y)
        goal_msg.radius=goal_rad
        goal_msg.angle_threshold=10.0
        
        # Send current goal
        goal_msg.region_center.pose=goal_point
        
        # Book Keeping
        goal_msg.region_center.header.frame_id="warty/map"  # Fixed frame in Rviz
        goal_msg.region_center.header.stamp=rospy.Time.now()
        # self.client.send_goal(goal_msg)
        self.client.send_goal(goal_msg, self.done_cb)

    def done_cb(self, status, result):
        self.goal_cnt += 1
        if status == 2:
            rospy.loginfo("Status 2 : Goal pose "+str(self.goal_cnt)+" received a cancel request after it started executing, completed execution!")
            return

        if status == 3:
            rospy.loginfo("Status 3 : Goal pose "+str(self.goal_cnt)+" reached") 
        
            self.client.cancel_all_goals()
            rospy.loginfo("Final goal pose reached!")    

            time.sleep(1)       # Adding wait time
            self.reached_goal=True
                
            return

        if status == 4:
            
            rospy.loginfo(" Status 4 : Goal pose "+str(self.goal_cnt)+" was aborted by the Action Server")
            # rospy.signal_shutdown("Goal pose "+str(self.goal_cnt)+" aborted, shutting down!")
            # return
            

            goal_point=Pose()
            
            # position
            goal_point.position.x= self.goal_x
            goal_point.position.y= self.goal_y
            goal_point.position.z= 0.0
            # Goal orientation
            goal_point.orientation.x= 0.0
            goal_point.orientation.y= 0.0
            goal_point.orientation.z= 0.6
            goal_point.orientation.w= 0.8

            goal_msg = nav_msg.GotoRegionGoal()
            # Radius and angle threshold
            goal_rad=self.determine_goal_radius(self.goal_x, self.goal_y)
            goal_msg.radius=goal_rad
            goal_msg.angle_threshold=10.0
            # Send current goal
            # goal_msg.region_center.pose.postion.x=self.frontier_x[goal_ind]
            # goal_msg.region_center.pose.postion.y=self.frontier_y[goal_ind]
            goal_msg.region_center.pose=goal_point
            # Book Keeping
            goal_msg.region_center.header.frame_id="warty/map"  # Fixed frame in Rviz
            goal_msg.region_center.header.stamp=rospy.Time.now()
            self.client.send_goal(goal_msg, self.done_cb)

            return
                

        if status == 5:
            rospy.loginfo(" Status 5 : Goal pose "+str(self.goal_cnt)+" has been rejected by the Action Server")
            # rospy.signal_shutdown("Goal pose "+str(self.goal_cnt)+" rejected, shutting down!")
            return

        if status == 8:
            rospy.loginfo("Status 8 : Goal pose "+str(self.goal_cnt)+" received a cancel request before it started executing, successfully cancelled!")
            return
            
    def do_recovery(self) :
        rospy.loginfo("Entering recovery mode")

        if len(self.path_x_coord) < 5:  # Need at least a few points for recovery
            rospy.logwarn("Not enough path history for recovery")
            return False
        
        pathx=self.path_x_coord
        pathy=self.path_y_coord

        ''' Consider n th pose from back'''

        recover_point_x=pathx[-2]
        recover_point_y=pathy[-2]

        current_pose = self.keyframe_poses[-1]


        path_angle=math.atan2((current_pose.y() - recover_point_y),(current_pose.x() - recover_point_x))
        diff_angle=path_angle - current_pose.theta()

        print('path angle', path_angle)
        print('current angle', current_pose.theta())
        print('difference angle', path_angle - current_pose.theta())

        # convert to 0 to 2pi
        diff_angle=diff_angle + 2*math.pi
        diff_angle=diff_angle % (2*math.pi)
        
        print("diff_angle ",diff_angle)

        vel_msg = Twist()

        # print("little going back")
        # vel_msg.linear.x=4.00
        # self.vel_pub.publish(vel_msg)
        # time.sleep(2)
        # vel_msg.linear.x=0.00
        # self.vel_pub.publish(vel_msg)
        # time.sleep(1)

        if ((0 <= diff_angle <= math.pi/2) or (3*math.pi/2 <= diff_angle <= 2*math.pi)):
    
            print("going back")
            vel_msg.linear.x = -4.00
        else: 
            print("going forward")
            vel_msg.linear.x = 4.00
        
        self.vel_pub.publish(vel_msg)

        vel_msg.linear.x = 0.00
        # time.sleep(2)
        time.sleep(4)
        self.vel_pub.publish(vel_msg)

    def determine_goal_radius(self,x_goal,y_goal):
        ''' Determine the radius of goal accrding to current pos by taking min of default radius and distance between robot and goal'''
        dist_goal=self.distance_calculator(x_goal,y_goal)
        radius = min(dist_goal/3, self.goal_radius) # To ensure robot does't reach goal prematurely
        print('original goal radius, distance to goal, new_goal radius',self.goal_radius,dist_goal,radius)
        return radius
    
    def distance_calculator(self,x_coords,y_coords):
        ''' Implementing simple euclidean distances here.
            For more complex continuous path cost 
            implementation refer information_gain_test.py'''
        current_pose = self.keyframe_poses[-1]
        robot_loc_x = current_pose.x()
        robot_loc_y = current_pose.y()

        distances= np.sqrt((x_coords-robot_loc_x)**2 + (y_coords-robot_loc_y)**2)

        return distances

    def save_to_csv(self):
        """
        Save key SLAM metrics to CSV files for post-processing analysis:
        - Trajectory estimates vs ground truth
        - Map entropy over time
        - Localization uncertainty (covariance) with average trace over time
        """
        
        # Create timestamp for unique filename
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
        # Create output directory
        output_dir = os.path.expanduser(f"/home/arl-sse/phoenix-r1-copy/src/behaviors/single_robot_infosplorer/{timestamp}_gtSLAM")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 1. Save trajectory and RMSE data
        trajectory_file = os.path.join(output_dir, "trajectory.csv")
        rospy.loginfo(f"Saving trajectory data to {trajectory_file}")
        
        with open(trajectory_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "keyframe_id", 
                "estimated_x", "estimated_y", "estimated_theta",
                "ground_truth_x", "ground_truth_y", "ground_truth_theta",
                "error_x", "error_y", "error_theta", "position_error"
            ])
            
            # Process trajectory data
            total_position_error = 0.0
            count = 0
            
            # Process all keyframes with both estimate and ground truth
            for i in range(min(len(self.keyframe_poses), len(self.keyframe_ground_truth))):
                if i >= self.current_pose_id:
                    continue  # Skip if beyond current pose ID
                    
                # Get estimated pose
                est_pose = self.keyframe_poses[i]
                est_x = est_pose.x()
                est_y = est_pose.y()
                est_theta = est_pose.theta()
                
                # Get ground truth pose
                if isinstance(self.keyframe_ground_truth[i], tuple):
                    # If ground truth is stored as (x, y) tuple
                    gt_x, gt_y = self.keyframe_ground_truth[i]
                    gt_theta = 0.0  # No theta in the tuple
                elif hasattr(self.keyframe_ground_truth[i], 'x') and callable(getattr(self.keyframe_ground_truth[i], 'x')):
                    # If ground truth is stored as a Pose2
                    gt_x = self.keyframe_ground_truth[i].x()
                    gt_y = self.keyframe_ground_truth[i].y()
                    gt_theta = self.keyframe_ground_truth[i].theta()
                else:
                    # Skip this entry if ground truth format is unknown
                    rospy.logwarn(f"Unknown ground truth format for keyframe {i}")
                    continue
                
                # Calculate position error
                error_x = est_x - gt_x
                error_y = est_y - gt_y
                error_theta = est_theta - gt_theta
                # Normalize theta error to [-pi, pi]
                while error_theta > np.pi:
                    error_theta -= 2*np.pi
                while error_theta < -np.pi:
                    error_theta += 2*np.pi
                    
                position_error = np.sqrt(error_x**2 + error_y**2)
                total_position_error += position_error
                count += 1
                
                writer.writerow([
                    i, est_x, est_y, est_theta, 
                    gt_x, gt_y, gt_theta,
                    error_x, error_y, error_theta, position_error
                ])
        
        # Calculate and log RMSE
        rmse = 0.0
        if count > 0:
            rmse = np.sqrt(total_position_error / count)
            rospy.loginfo(f"Overall RMSE: {rmse:.4f} meters")
        
        # 2. Save map entropy over time
        entropy_file = os.path.join(output_dir, "map_entropy.csv")
        rospy.loginfo(f"Saving map entropy data to {entropy_file}")
        
        with open(entropy_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["keyframe_id", "map_entropy"])
            
            # Use the attribute name correctly (total_map_entropy vs total_map_entorpy)
            attr_name = 'total_map_entropy'
            if hasattr(self, 'total_map_entorpy'):  # Handle potential typo in attribute name
                attr_name = 'total_map_entorpy'
                
            # If we have entropy stored per keyframe
            if hasattr(self, 'entropy_history') and len(self.entropy_history) > 0:
                for i, entropy in enumerate(self.entropy_history):
                    writer.writerow([i, entropy])
            else:
                # If we only have the current entropy
                if hasattr(self, attr_name):
                    writer.writerow([self.current_pose_id - 1, getattr(self, attr_name)])
        
        # 3. Save covariance/uncertainty data from GTSAM
        uncertainty_file = os.path.join(output_dir, "localization_uncertainty.csv")
        rospy.loginfo(f"Saving localization uncertainty data to {uncertainty_file}")
        
        # Initialize arrays to track average covariance trace over time
        pose_traces = []
        position_traces = []  # For just X,Y position covariance
        orientation_variances = []  # For just theta variance
        
        with open(uncertainty_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "keyframe_id", "cov_xx", "cov_xy", "cov_xt", 
                "cov_yx", "cov_yy", "cov_yt",
                "cov_tx", "cov_ty", "cov_tt",
                "position_trace", "full_trace", "determinant", "max_eigenvalue",
                "avg_position_trace_so_far", "avg_full_trace_so_far",
                "avg_orientation_variance_so_far"
            ])
            
            # Try to compute pose covariances from the factor graph
            try:
                # Create full copy of the internal ISAM2 Bayes tree factor graph
                full_graph = self.isam.getFactorsUnsafe() if hasattr(self, 'isam') else self.graph
                
                # Create marginals from the graph
                marginals = gtsam.Marginals(full_graph, self.current_estimate)
                
                for i in range(self.current_pose_id):
                    if self.current_estimate.exists(i):
                        try:
                            # Get full covariance matrix for this pose
                            cov_matrix = marginals.marginalCovariance(i)
                            
                            # Extract values from the 3x3 covariance matrix
                            cov_xx = cov_matrix[0, 0]
                            cov_xy = cov_matrix[0, 1]
                            cov_xt = cov_matrix[0, 2]
                            cov_yx = cov_matrix[1, 0]
                            cov_yy = cov_matrix[1, 1]
                            cov_yt = cov_matrix[1, 2]
                            cov_tx = cov_matrix[2, 0]
                            cov_ty = cov_matrix[2, 1]
                            cov_tt = cov_matrix[2, 2]
                            
                            # Compute derived metrics
                            position_trace = cov_xx + cov_yy  # Trace of position covariance
                            full_trace = np.trace(cov_matrix)  # Full trace including orientation
                            determinant = np.linalg.det(cov_matrix)
                            
                            # Store for average calculation
                            pose_traces.append(full_trace)
                            position_traces.append(position_trace)
                            orientation_variances.append(cov_tt)
                            
                            # Compute running averages
                            avg_full_trace = sum(pose_traces) / len(pose_traces)
                            avg_position_trace = sum(position_traces) / len(position_traces)
                            avg_orientation_var = sum(orientation_variances) / len(orientation_variances)
                            
                            # Compute eigenvalues
                            eigenvalues = np.linalg.eigvals(cov_matrix)
                            max_eigenvalue = np.max(np.abs(eigenvalues))
                            
                            writer.writerow([
                                i, cov_xx, cov_xy, cov_xt, 
                                cov_yx, cov_yy, cov_yt,
                                cov_tx, cov_ty, cov_tt,
                                position_trace, full_trace, determinant, max_eigenvalue,
                                avg_position_trace, avg_full_trace, avg_orientation_var
                            ])
                        except Exception as e:
                            rospy.logwarn(f"Could not compute covariance for pose {i}: {e}")
                            writer.writerow([
                                i, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                            ])
            except Exception as e:
                rospy.logwarn(f"Could not compute marginals: {e}")
                # If we have stored covariance matrices in self.keyframe_poses_cov
                if hasattr(self, 'keyframe_poses_cov'):
                    for i, cov_matrix in enumerate(self.keyframe_poses_cov):
                        if i < self.current_pose_id:
                            try:
                                # Process covariance matrix
                                cov_xx = cov_matrix[0, 0]
                                cov_xy = cov_matrix[0, 1]
                                cov_xt = cov_matrix[0, 2] if cov_matrix.shape[0] > 2 else 0.0
                                cov_yx = cov_matrix[1, 0]
                                cov_yy = cov_matrix[1, 1]
                                cov_yt = cov_matrix[1, 2] if cov_matrix.shape[0] > 2 else 0.0
                                cov_tx = cov_matrix[2, 0] if cov_matrix.shape[0] > 2 else 0.0
                                cov_ty = cov_matrix[2, 1] if cov_matrix.shape[0] > 2 else 0.0
                                cov_tt = cov_matrix[2, 2] if cov_matrix.shape[0] > 2 else 0.0
                                
                                position_trace = cov_xx + cov_yy
                                full_trace = np.trace(cov_matrix)
                                
                                # Store for average calculation
                                pose_traces.append(full_trace)
                                position_traces.append(position_trace)
                                orientation_variances.append(cov_tt)
                                
                                # Calculate running averages
                                avg_full_trace = sum(pose_traces) / len(pose_traces)
                                avg_position_trace = sum(position_traces) / len(position_traces)
                                avg_orientation_var = sum(orientation_variances) / len(orientation_variances)
                                
                                determinant = np.linalg.det(cov_matrix)
                                eigenvalues = np.linalg.eigvals(cov_matrix)
                                max_eigenvalue = np.max(np.abs(eigenvalues))
                                
                                writer.writerow([
                                    i, cov_xx, cov_xy, cov_xt, 
                                    cov_yx, cov_yy, cov_yt,
                                    cov_tx, cov_ty, cov_tt,
                                    position_trace, full_trace, determinant, max_eigenvalue,
                                    avg_position_trace, avg_full_trace, avg_orientation_var
                                ])
                            except Exception as inner_e:
                                rospy.logwarn(f"Error processing covariance for pose {i}: {inner_e}")
                                writer.writerow([
                                    i, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                                ])
        
        # Save average covariance trace statistics to a separate file
        avg_trace_file = os.path.join(output_dir, "average_covariance_trace.csv")
        rospy.loginfo(f"Saving average covariance trace data to {avg_trace_file}")
        
        with open(avg_trace_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["metric", "value"])
            
            if position_traces:
                avg_position_trace = sum(position_traces) / len(position_traces)
                writer.writerow(["average_position_covariance_trace", avg_position_trace])
                
                # Also calculate statistics over time segments to see trends
                if len(position_traces) >= 3:
                    early_segment = position_traces[:len(position_traces)//3]
                    middle_segment = position_traces[len(position_traces)//3:2*len(position_traces)//3]
                    late_segment = position_traces[2*len(position_traces)//3:]
                    
                    writer.writerow(["early_position_covariance_trace", sum(early_segment)/len(early_segment)])
                    writer.writerow(["middle_position_covariance_trace", sum(middle_segment)/len(middle_segment)])
                    writer.writerow(["late_position_covariance_trace", sum(late_segment)/len(late_segment)])
            
            if pose_traces:
                avg_full_trace = sum(pose_traces) / len(pose_traces)
                writer.writerow(["average_full_covariance_trace", avg_full_trace])
            
            if orientation_variances:
                avg_orientation_var = sum(orientation_variances) / len(orientation_variances)
                writer.writerow(["average_orientation_variance", avg_orientation_var])
        
        # 4. Write a summary file with key results
        summary_file = os.path.join(output_dir, "summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Graph SLAM Results Summary\n")
            f.write(f"========================\n\n")
            f.write(f"Run timestamp: {timestamp}\n")
            f.write(f"Total keyframes: {self.current_pose_id}\n")
            f.write(f"Total poses in graph: {len(self.keyframe_poses)}\n")
            f.write(f"RMSE: {rmse:.4f} meters\n")
            
            # Include map entropy
            attr_name = 'total_map_entropy' if hasattr(self, 'total_map_entropy') else 'total_map_entorpy'
            if hasattr(self, attr_name):
                f.write(f"Final map entropy: {getattr(self, attr_name):.4f}\n")
                
            # Include loop closure count if available
            if hasattr(self, 'loop_closure_count'):
                f.write(f"Loop closures detected: {self.loop_closure_count}\n")
            
            # Add average covariance trace results
            if position_traces:
                f.write(f"Average position covariance trace: {avg_position_trace:.6f}\n")
            if pose_traces:
                f.write(f"Average full covariance trace: {avg_full_trace:.6f}\n")
            if orientation_variances:
                f.write(f"Average orientation variance: {avg_orientation_var:.6f}\n")
                
            # Summary statistics from the final covariance
            try:
                marginals = gtsam.Marginals(full_graph, self.current_estimate)
                if self.current_pose_id > 0 and self.current_estimate.exists(self.current_pose_id - 1):
                    latest_cov = marginals.marginalCovariance(self.current_pose_id - 1)
                    f.write(f"Final position uncertainty (trace): {np.trace(latest_cov[:2,:2]):.6f}\n")
                    f.write(f"Final orientation uncertainty: {latest_cov[2,2]:.6f}\n")
            except Exception as e:
                f.write(f"Could not compute final covariance: {e}\n")
        
        # Save backward compatibility file
        if hasattr(self, 'output_file'):
            with open(self.output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["current_id", "Estimated_x", "Estimated_y", "Ground_Truth_x", "Ground_Truth_y", "Covariance_Trace", "Map_Entropy"])
                
                # Create simplified data rows for backward compatibility
                for i in range(min(len(self.keyframe_poses), len(self.keyframe_ground_truth))):
                    if i < self.current_pose_id:
                        est_pose = self.keyframe_poses[i]
                        est_x = est_pose.x()
                        est_y = est_pose.y()
                        
                        # Get ground truth data
                        if isinstance(self.keyframe_ground_truth[i], tuple):
                            gt_x, gt_y = self.keyframe_ground_truth[i]
                        else:
                            gt_x = self.keyframe_ground_truth[i].x()
                            gt_y = self.keyframe_ground_truth[i].y()
                        
                        # Get covariance trace if available
                        cov_trace = position_traces[i] if i < len(position_traces) else 0.0
                        
                        # Get map entropy if available
                        entropy = self.entropy_history[i] if hasattr(self, 'entropy_history') and i < len(self.entropy_history) else getattr(self, attr_name, 0.0)
                        
                        writer.writerow([i, est_x, est_y, gt_x, gt_y, cov_trace, entropy])
        
        rospy.loginfo(f"All SLAM metrics saved to {output_dir}")
        return output_dir

    def behavioral_entropy_calculator(self,probs):
        p=np.copy(probs)
        entropy=np.zeros(np.shape(p))
        w1=np.zeros(np.shape(p))
        w2=np.zeros(np.shape(p))
        p_not_zero=p[p!=0]
        # w1=np.exp(-self.beta*np.power((-np.log(p_not_zero)),self.alpha))
        # w2=np.exp(-self.beta*np.power((-np.log(1-p_not_zero)),self.alpha))

        w1[p!=0]=np.exp(-self.beta*np.power((-np.log(p_not_zero)),self.alpha))
        w2[p!=0]=np.exp(-self.beta*np.power((-np.log(1-p_not_zero)),self.alpha))

        entropy[(w1!=0)&(w2!=0)]=-w1[(w1!=0)&(w2!=0)]*np.log(w1[(w1!=0)&(w2!=0)]) -w2[(w1!=0)&(w2!=0)]*np.log(w2[(w1!=0)&(w2!=0)])

        # return entropy
        return np.sum(entropy)
    
    def shannon_entropy_calculator(self,probs):
        ''' takes an array of bernoulli trials and returns the entropy for each trail'''
        # p=np.copy(probs)
        # entropy=np.zeros(np.shape(p))
        # p_not_zero=p[p!=0]
        # entropy[p!=0]=-p_not_zero*np.log(p_not_zero) -(1-p_not_zero)*np.log(1-p_not_zero) # Calculate Shannons entropy
        # # return entropy
        # return np.sum(entropy)
    
        p = np.copy(probs)
        entropy = np.zeros(np.shape(p))
        epsilon = 1e-6  # Small value to handle log(0)
        p = np.clip(p, epsilon, 1 - epsilon)  # Avoid exact 0 or 1
        entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)  # Calculate Shannon's entropy
        return np.sum(entropy)
    
    def renyi_entropy_calculator(self,probs):
        p=np.copy(probs)
        entropy=np.zeros(np.shape(p))
        p_not_zero=p[p!=0]
        entropy[p!=0]=(1/(1-self.renyi))*np.log(np.power(p_not_zero,self.renyi) + np.power((1-p_not_zero),self.renyi))
        return entropy

if __name__=="__main__":

    try:
        navigator = graph_active_slam()

    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation interrupted")
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred: {e}")
    finally:
        rospy.loginfo("Saving data to CSV...")
        try:
            navigator.save_to_csv()
        except Exception as save_error:
            rospy.logerr(f"Failed to save data: {save_error}")
        else:
            rospy.loginfo("Data saved successfully")  