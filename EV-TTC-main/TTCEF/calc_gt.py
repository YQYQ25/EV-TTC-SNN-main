"""
T^2CEF Ground Truth Calculator for Event Camera Data.

This script processes M3ED event camera data to generate ground truth TTC,
optical flow, and depth information. It uses camera poses and depth data
to calculate TTC values and flow vectors between consecutive frames.

The script handles camera calibration, pose interpolation, and renders
the results using datashader for proper splatting of sparse depth points.

Usage:
    python calc_gt.py --num 0 --data_dir /path/to/data --dt 0.2 --splat 3 --full_res False
"""

import argparse
import os
import sys
import time

import cv2
import datashader as ds
import datashader.transfer_functions as tf
import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import expm, logm
from tqdm import tqdm
from projectaria_tools.core.sophus import SE3, interpolate
from typing import Tuple, List, Dict, Any, Optional
S_TO_MS=1e6
def load_camera_params(f_data: h5py.File) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and process camera calibration parameters.
    
    Args:
        f_data: HDF5 file containing camera parameters
        
    Returns:
        x_map: X-coordinates map for undistortion
        y_map: Y-coordinates map for undistortion
        K: Camera intrinsic matrix (adjusted for undistortion)
    """
    # Load camera parameters from file
    D = f_data["prophesee"]["left"]["calib"]["distortion_coeffs"][:]  # Distortion coefficients
    intr = np.array(f_data["prophesee"]["left"]["calib"]["intrinsics"][:])  # Intrinsic parameters
    K = np.array([[intr[0], 0, intr[2]], [0, intr[1], intr[3]], [0, 0, 1]])  # Construct intrinsic matrix
    res = np.array([720, 1280])  # Camera resolution (height, width)
    
    # Calculate optimal camera matrix for undistortion
    new_mtx, p_roi = cv2.getOptimalNewCameraMatrix(K, D, res[::-1], 0)  # (width, height) format for OpenCV
    R = np.eye(3)  # No rotation applied (identity matrix)
    
    # Generate undistortion maps
    x_map, y_map = cv2.initUndistortRectifyMap(
        K, D, R, new_mtx, res[::-1], cv2.CV_32FC1
    )
    K = new_mtx.astype(np.float32)  # Use the updated camera matrix
    
    return x_map, y_map, K

def generate_poses(f_pose: h5py.File, time: np.ndarray, ) -> np.ndarray:
    """
    Generate interpolated camera poses for specified timestamps.
    
    Args:
        f_pose: HDF5 file containing camera poses
        time: Array of timestamps to interpolate poses for
        
    Returns:
        np.ndarray: Array of interpolated transformation matrices
    """
    # Extract poses and timestamps from file
    pose = f_pose["Cn_T_C0"][:].astype(np.float32)  # All camera poses
    pose_time = f_pose["ts"][:].astype(np.float32)  # Corresponding timestamps
    
    # Convert matrices to Sophus SE3 objects for proper interpolation
    soph_pose = [SE3.from_matrix(pose[i]) for i in range(len(pose))]
    
    # Find nearest pose indices for each timestamp
    start_pose_idx = np.searchsorted(pose_time, time)
    
    interp_pose = []
    for i in range(len(time)):
        # Get timestamps for bracketing poses
        ts_start = pose_time[start_pose_idx[i]-1]
        ts_end = pose_time[start_pose_idx[i]]
        t_query = time[i]
        
        # Calculate interpolation factor (0 to 1)
        alpha = (t_query - ts_start) / (ts_end - ts_start)
        
        # Perform pose interpolation using Sophus library
        interp_pose.append(np.array(
            interpolate(
                soph_pose[start_pose_idx[i]-1],
                soph_pose[start_pose_idx[i]],
                alpha
            ).to_matrix()
        ))
    return np.stack(interp_pose)


def convert_file(data_dir: str, seq_name: str, dt: float, splat: int, full_res: bool):
    """
    Process a sequence file to generate TTC ground truth data.
    
    Args:
        data_dir: Base directory containing the dataset
        seq_name: Name of the sequence to process
        dt: Time interval for optical flow calculation
        splat: Number of pixels to splat for depth points
        full_res: Whether to use full resolution or crop to 360x360
    """
    print(f"Processing sequence: {seq_name}")
    # Create and Load M3ED, Events, Pose and Exp Filters Files
    data_path = f"{data_dir}/events/m3ed/{seq_name}/{seq_name}_data.h5"
    pose_path = f"{data_dir}/depth/m3ed/{seq_name}/{seq_name}_pose_gt.h5"
    depth_path = f"{data_dir}/depth/m3ed/{seq_name}/{seq_name}_depth_gt.h5"
    exp_path = f"{data_dir}/exp_filts/m3ed/{seq_name}.h5"

    # Open all necessary HDF5 files
    f_data = h5py.File(data_path, "r")
    f_pose = h5py.File(pose_path, "r")
    f_exp = h5py.File(exp_path, "r")
    f_depth = h5py.File(depth_path, "r")

    # Load camera calibration parameters
    x_map, y_map, K = load_camera_params(f_data)

    # Get pose and depth timestamps for boundary checking
    pose_times = f_pose['ts'][:].astype(np.float32)
    depth_times = f_depth['ts'][:].astype(np.float32)
    
    # Get exposure times and check boundaries
    all_exp_times = f_exp['exp_times'][:].astype(np.float32)
    
    # Ensure exposure times are within pose and depth time boundaries
    min_pose_time = pose_times[1]  # Use second time point as pose interpolation requires previous point
    max_pose_time = pose_times[-1]
    min_depth_time = depth_times[0]
    max_depth_time = depth_times[-1]
    
    # Filter exposure times to be within valid bounds
    # For start times, ensure they're within depth and pose bounds, considering dt offset
    valid_exp_mask = (
        (all_exp_times >= min_pose_time) & 
        (all_exp_times <= max_pose_time) & 
        (all_exp_times >= min_depth_time) &
        (all_exp_times <= max_depth_time) &
        ((all_exp_times - dt*S_TO_MS) >= min_pose_time)  # Start time must be valid too
    )
    
    

    end_times = all_exp_times

    print(f"Using {np.sum(valid_exp_mask)} out of {len(all_exp_times)} exposure frames after boundary checks")
    # Calculate start times based on filtered end times
    start_times = end_times - dt*S_TO_MS  # Start times (shifted by dt)
    
    # Generate interpolated poses for start and end times
    CN_T_C0 = generate_poses(f_pose, start_times)  # Start pose relative to reference frame
    CNP1_T_C0 = generate_poses(f_pose, end_times)  # End pose relative to reference frame

    # Calculate relative transformation from start to end pose
    CN_T_CNP1 = np.stack([CN_T_C0[i] @ np.linalg.inv(CNP1_T_C0[i]) for i in range(len(CN_T_C0))])
    T = CN_T_CNP1[:, :3, -1]/dt  # Extract translational velocity components

    # Calculate rotational velocities using matrix logarithm
    w_ls = [
        logm(CN_T_CNP1[i, :3, :3], disp=False)[0]
        for i in range(len(CN_T_CNP1))
    ]
    
    # Extract angular velocity components from skew-symmetric matrices
    Omega = np.stack(
        [np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]]) for w_hat in w_ls]
    )/dt
    

    # Load depth data
    depth = f_depth['depth']['prophesee']['left']
    depth_ts = f_depth['ts'][:]
    depth_pose = f_depth['Cn_T_C0'][:]
    
    # Find closest depth frames to our start
    
    start_depth = np.searchsorted(depth_ts, start_times)

    # Create projection points for undistortion (in camera coordinate system)
    # Depth points are in distorted frame: https://github.com/daniilidis-group/m3ed/blob/df739f20fba41ac6da8c22f4260c305875e391ed/build_system/lidar_depth/fasterlio_gt.py#L61
    proj_pts = np.linalg.inv(K) @ np.stack(
        [x_map.flatten(), y_map.flatten(), np.ones_like(x_map.flatten())]
    )
    proj_pts /= proj_pts[2]  # Normalize to get unit z

    # Create output directory if it doesn't exist
    output_dir = f"{data_dir}/ttcef/m3ed"
    os.makedirs(output_dir, exist_ok=True)

    # Create output HDF5 file
    output_file = f"{output_dir}/{seq_name}.h5"
    f_w = h5py.File(output_file, "w")

    # Determine dimensions based on full_res flag
    res = (720, 1280) if full_res else (360, 360)
    
    # Create compressed datasets with appropriate dimensions
    # Using Blosc2 compression with LZ4 algorithm for good compression/speed balance
    depth_dset = f_w.create_dataset("depth", 
                      shape=(len(start_times), res[0], res[1]),
                      dtype=np.float32,
                      chunks=(1, res[0], res[1]),
                      **hdf5plugin.Blosc2(cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE))
    
    ttc_dset = f_w.create_dataset("ttc", 
                    shape=(len(start_times), res[0], res[1]),
                    dtype=np.float32,
                    chunks=(1, res[0], res[1]),
                    **hdf5plugin.Blosc2(cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE))
    
    flow_dset = f_w.create_dataset("flow", 
                     shape=(len(start_times), 2, res[0], res[1]),
                     dtype=np.float32,
                     chunks=(1, 2, res[0], res[1]),
                     **hdf5plugin.Blosc2(cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE))
    
    mask_dset = f_w.create_dataset("mask", 
                     shape=(len(start_times), res[0], res[1]),
                     dtype=bool,
                     chunks=(1, res[0], res[1]),
                     **hdf5plugin.Blosc2(cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE))
    
    valid=valid_exp_mask
    # Create metadata datasets
    f_w.create_dataset("T", data=T, dtype=np.float32)  # Translation velocity
    f_w.create_dataset("Omega", data=Omega, dtype=np.float32)  # Rotational velocity
    f_w.create_dataset("NP_T_NP1", data=CN_T_CNP1, dtype=np.float32)  # Relative transformation
    f_w.create_dataset("ts", data=start_times)  # Start timestamps
    f_w.create_dataset("te", data=end_times)  # End timestamps
    f_w.create_dataset("K", data=K)  # Camera intrinsics
    f_w.create_dataset("ev_file", data=data_path)  # Source file paths
    f_w.create_dataset("pose_file", data=pose_path)
    f_w.create_dataset("depth_file", data=depth_path)
    f_w.create_dataset("valid", data=valid)

    def process_idx(idx: int):
        """
        Process a single frame to generate depth, TTC, flow and mask.
        
        Args:
            idx: Frame index to process
            
        Returns:
            tuple: Processed frame data (idx, depth_img, ttc_img, flow_data, mask_data)
        """
        # Get depth to event transform
        if not valid_exp_mask[idx]:
            depth_img = np.zeros((res[0], res[1]), dtype=np.float32)
            ttc = np.zeros((res[0], res[1]), dtype=np.float32)
            flow = np.zeros((2, res[0], res[1]), dtype=np.float32)
            mask = np.zeros((res[0], res[1]), dtype=bool)
            return idx, depth_img, ttc, flow, mask
        depth_raw = depth[int(start_depth[idx])]  # Closest depth frame to current time
        Dn_T_C0 = depth_pose[start_depth[idx]]  # Depth frame pose
        CN_T_DN = CN_T_C0[idx] @ np.linalg.inv(Dn_T_C0)  # Transform from depth to current event frame
        
        # Transform depth data to current event frame
        flat_depth = depth_raw.flatten()
        valid_pts = ~np.isinf(flat_depth)  # Filter out invalid depth points
        valid_depth = flat_depth[valid_pts]

        # Project depth points into 3D space
        depth_cord_N = proj_pts[:, valid_pts] * valid_depth[None, :]
        depth_cord_N_aug = np.vstack(
            [depth_cord_N, np.ones_like(depth_cord_N[0])]
        )
        
        # Transform depth points to event frame
        event_cord_N = CN_T_DN @ depth_cord_N_aug
        
        # Project 3D points back to image plane
        img_cord_N = K @ event_cord_N[:3]
        img_cord_N /= img_cord_N[2]  # Normalize by Z

        if not full_res:
            # Crop central region when using reduced resolution
            crop_pts = (
                    (img_cord_N[0] > 280)
                    & (img_cord_N[0] < 1000)
                    & (img_cord_N[1] > 0)
                    & (img_cord_N[1] < 720)
                )
            
            img_cord_N = img_cord_N[:, crop_pts]
            
            # Adjust coordinates for cropped region
            img_cord_N[0] -= 280
            img_cord_N /= 2  # Scale down by 2x
            depth_splat = valid_depth[crop_pts]
        else:
            depth_splat = valid_depth
        
        # Filter out negative depth values
        pos_depth_idx = depth_splat > 0
        
        # Use datashader to splat depth points into a depth map
        df = pd.DataFrame(
            {"x": img_cord_N[0][pos_depth_idx], "y": img_cord_N[1][pos_depth_idx], "d": depth_splat[pos_depth_idx]}
        )
        canvas = ds.Canvas(
            plot_width=res[1],
            plot_height=res[0],
            x_range=(0, res[1]),
            y_range=(0, res[0]),
            x_axis_type="linear",
            y_axis_type="linear",
        )
        pt = canvas.points(df, "x", "y", agg=ds.min("d"))  # Use minimum aggregation for depth
        depth_img = tf.spread(pt, px=splat, how="min").values  # Spread points by splat pixels
        
        # Calculate TTC: depth divided by velocity in Z direction (with small epsilon to avoid division by zero)
        ttc = depth_img / (T[idx, 2] + 1e-5)

        # Prepare camera matrix for flow calculation
        if not full_res:
            K_flow = K.copy()
            K_flow[0, 2] -= 280  # Adjust principal point for cropping
            K_flow /= 2  # Scale for reduced resolution
            K_flow[2, 2] = 1  # Ensure last element is 1
        else:
            K_flow = K.copy()
            
        # Calculate optical flow using only points with valid depth values
        yy, xx = np.where(~np.isnan(depth_img))
        cam_pts = np.stack([xx, yy, np.ones_like(xx)], axis=1)
        norm_pts = np.linalg.inv(K_flow) @ cam_pts.T
        cam_cords_N = norm_pts * depth_img[yy, xx][None, :]
        cam_cords_N = np.vstack(
            [cam_cords_N, np.ones_like(cam_cords_N[0])]
        )

        # Calculate the optical flow by transforming points to next frame
        event_cord_NP1 = np.linalg.inv(CN_T_CNP1[idx]) @ cam_cords_N
        img_cord_NP1 = K_flow @ event_cord_NP1[:3]
        img_cord_NP1 /= img_cord_NP1[2]  # Normalize by Z

        # Flow is the difference between projected coordinates
        flow_pt = img_cord_NP1 - cam_pts.T 
   
        # Create dense flow field (sparse-to-dense conversion)
        flow = np.zeros((2, res[0], res[1])) # We are just qunatizing here, you can splat similary to depth
        flow[0][cam_pts[:, 1].astype(np.uint16), cam_pts[:, 0].astype(np.uint16)] = flow_pt[0]
        flow[1][cam_pts[:, 1].astype(np.uint16), cam_pts[:, 0].astype(np.uint16)] = flow_pt[1]
        
        # Create mask for valid depth/flow values
        mask = ~np.isnan(depth_img)
        
        return idx, depth_img, ttc, flow, mask.astype(np.uint8)
    
    def process_and_save(idx: int):
        """
        Process a frame and save results directly to the HDF5 file.
        
        Args:
            idx: Frame index to process
            
        Returns:
            int: Processed frame index
        """
        result = process_idx(idx)
        idx, depth_img, ttc_img, flow_data, mask_data = result
        
        # Each worker writes to its own index, no lock needed
        depth_dset[idx] = depth_img
        ttc_dset[idx] = ttc_img
        flow_dset[idx] = flow_data
        mask_dset[idx] = mask_data
        
        # Return the index to track progress
        return idx

    # Process data and save results concurrently using joblib
    results = Parallel(n_jobs=12, backend="threading")(
        delayed(process_and_save)(i) for i in tqdm(range(len(start_times)))
    )
    # results = Parallel(n_jobs=12, backend="threading")(
    #     delayed(process_and_save)(i) for i in tqdm(range(3000))
    # )
    
    # Close the file after all processing is done
    f_w.close()
    print(f"Data saved to {output_file}")

def run(args: argparse.Namespace) -> None:
    """
    Process a sequence file based on command-line arguments.
    
    Args:
        args: Command-line arguments
    """
    m3ed_directory = args.data_dir
    
    try:
        # Get list of available sequences
        files_ls = os.listdir(m3ed_directory + "/events/m3ed/")
    except FileNotFoundError:
        print(f"Error: Directory {m3ed_directory} not found.")
        sys.exit(1)
        
    if not files_ls:
        print("No matching sequence files found.")
        return
    
    files_ls.sort()

    try:
        # Process the specified sequence
        convert_file(args.data_dir, files_ls[args.num].split('.h5')[0], args.dt, args.splat, args.full_res)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"Error processing file {files_ls[args.num]} ({args.num}): {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process event camera data to calculate time-to-contact ground truth."
    )
    parser.add_argument("--num", help="File number to process", type=int, default=0)
    parser.add_argument("--data_dir", help="Directory of M3ED Event Folders", type=str, default="/tmp/")
    parser.add_argument("--dt", help="Period for Optical Pred exp_time-dt in sec", type=float, default=0.010)
    parser.add_argument("--splat", help="Number of pixels to splat depth points", type=int, default=3)
    parser.add_argument("--full_res", help="Use full resolution (720x1280) instead of 360x360", action='store_true')
    args = parser.parse_args()
    run(args)
