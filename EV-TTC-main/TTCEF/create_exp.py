#!/usr/bin/env python3
"""
Exponential Filter Event Processing Module

This module processes event camera data from the M3ED dataset and applies
exponential filters to generate event representation. This code is meant
to mirror the C++ code applied online to the events in real-time.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import cv2
import h5py
import hdf5plugin
import numba
import numpy as np
from tqdm import tqdm

# Configuration
MS_TO_S=1e-3
US_TO_S=1e-6
def calc_dist_map_ds(x_map: np.ndarray, y_map: np.ndarray) -> np.ndarray:
    """
    Calculate distance mapping for downsampled image.
    
    Args:
        x_map: X-coordinate mapping from camera calibration
        y_map: Y-coordinate mapping from camera calibration
        
    Returns:
        dist_map: Distance map for energy sharing between pixels
    """
    dist_map = np.zeros((720, 720, 6), dtype=np.float32)
    for y in range(720):
        for x in range(280, 1000):
            id1 = (x_map[y, x] / 2) - 140 # Downsampled x-coordinate
            id2 = y_map[y, x] / 2 # Downsampled y-coordinate
            if id1 < 0 or id2 < 0 or id1 > 359 or id2 > 359: # Out of bounds check
                continue
            nc1 = int(id1)
            nc2 = int(id2)
            dist_map[y, x - 280, 0] = nc1
            dist_map[y, x - 280, 1] = nc2
            dist_map[y, x - 280, 2] = (1 - (id1 - nc1)) * (1 - (id2 - nc2)) # Energy Ratios for interpolation
            dist_map[y, x - 280, 3] = (1 - (nc1 + 1 - id1)) * (1 - (id2 - nc2))
            dist_map[y, x - 280, 4] = (1 - (id1 - nc1)) * (1 - (nc2 + 1 - id2))
            dist_map[y, x - 280, 5] = (1 - (nc1 + 1 - id1)) * (1 - (nc2 + 1 - id2))
    return dist_map


def calc_filt_constants(alphas: np.ndarray, time_bins: int) -> np.ndarray:
    """
    Calculate filter constants for exponential decay.
    
    (These are Algo. 2 Coefficients in the paper)
    Args:
        alphas: Array of alpha values for different decay rates
        time_bins: Number of time bins to calculate constants for
        
    Returns:
        filt_constants: Filter constants for each alpha and time bin
    """
    filt_constants = np.zeros((len(alphas), time_bins + 1), dtype=np.float32)
    for i in range(len(alphas)):
        for j in range(time_bins):
            filt_constants[i, j] = alphas[i] * ((1 - alphas[i]) ** -j)
        filt_constants[i, time_bins] = (1 - alphas[i]) ** time_bins

    return filt_constants


@numba.njit(cache=True)
def calc_exp(x_ev: np.ndarray, y_ev: np.ndarray, p_ev: np.ndarray, 
             t_ev: np.ndarray, dist_map: np.ndarray, 
             filt_constants: np.ndarray, exp_img: np.ndarray, 
             last_active: float, frame_interval:float = 0.007, time_bin=0.0002) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Calculate exponential filter representation from event data.
    Mirror of C++ code applied online to the events in real-time.

    Args:
        x_ev: X-coordinates of events
        y_ev: Y-coordinates of events
        p_ev: Polarities of events
        t_ev: Timestamps of events
        dist_map: Distance map for interpolation
        filt_constants: Filter constants for decay
        exp_img: Current exponential filter image
        last_active: Timestamp of last active event
        
    Returns:
        save_imgs: Array of saved filter images
        exp_img: Updated exponential filter image
        exp_times: Timestamps for saved images
        last_active: Updated timestamp of last active event
    """

    total_filts = int((t_ev[-1] - t_ev[0]) * US_TO_S // frame_interval) + 1
    save_imgs = np.zeros((total_filts, len(filt_constants), 360, 360), dtype=np.float32)
    exp_times = np.zeros((total_filts))
    cnt = 0
    
    for i in range(len(x_ev)):
        time_since_last = (t_ev[i] - last_active) * US_TO_S

        if time_since_last > frame_interval:
            last_active = t_ev[i]
            time_since_last = (t_ev[i] - last_active) * US_TO_S
            exp_img = exp_img * filt_constants[:, -1][:, None, None]

            exp_times[cnt] = last_active
            save_imgs[cnt] = exp_img
            cnt += 1

        update = dist_map[y_ev[i], x_ev[i] - 280]

        x_c = int(update[0])
        y_c = int(update[1])

        act_ind = int(time_since_last / time_bin)
        sign = 1 if p_ev[i] else -1

        for j in range(len(exp_img)):
            exp_img[j, y_c, x_c] += filt_constants[j][act_ind] * sign * update[2]
            exp_img[j, y_c, x_c + 1] += filt_constants[j][act_ind] * sign * update[3]
            exp_img[j, y_c + 1, x_c + 1] += filt_constants[j][act_ind] * sign * update[5]
            exp_img[j, y_c + 1, x_c] += filt_constants[j][act_ind] * sign * update[4]

    return save_imgs[:cnt], exp_img, exp_times[:cnt], last_active


def load_camera_params(f_data: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and process camera calibration parameters.
    
    Args:
        f_data: HDF5 file containing camera parameters
        
    Returns:
        dist_map: Calculated distance map
        res: Camera resolution
    """
    # Load camera parameters
    D = f_data["prophesee"]["left"]["calib"]["distortion_coeffs"][:]
    intr = np.array(f_data["prophesee"]["left"]["calib"]["intrinsics"][:])
    K = np.array([[intr[0], 0, intr[2]], [0, intr[1], intr[3]], [0, 0, 1]])
    res = np.array([720, 1280])

    # Undistortion setup
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(K, D, res[::-1], 0)
    R = np.eye(3)
    x_map, y_map = cv2.initUndistortRectifyMap(
        K, D, R, new_mtx, res[::-1], cv2.CV_32FC1
    )
    
    # Calculate distance map for downsampled image
    dist_map = calc_dist_map_ds(x_map, y_map)
    
    return dist_map, res


def process_event_batch(f_data: h5py.File, batch_idx: int, batch_size: int, 
                         dist_map: np.ndarray, filt_constants: np.ndarray, 
                         exp_img: np.ndarray, last_active: float, frame_interval:float, time_bin:float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Process a batch of events.
    
    Args:
        f_data: HDF5 file containing event data
        batch_idx: Index of the current batch
        batch_size: Size of each batch
        dist_map: Distance map for energy sharing
        filt_constants: Filter constants for decay
        exp_img: Current exponential filter image
        last_active: Timestamp of last active event
        
    Returns:
        save_exp: Array of saved filter images
        exp_img: Updated exponential filter image
        exp_times: Timestamps for saved images
        last_active: Updated timestamp of last active event
    """
    ms_map = f_data["prophesee"]["left"]["ms_map_idx"][:]
    start_idx = ms_map[batch_idx * batch_size]
    end_idx = ms_map[(batch_idx + 1) * batch_size]
    
    x_ev = f_data["prophesee"]["left"]["x"][start_idx:end_idx]
    
    # Filter events within valid x-coordinate range
    act_ind = ~((x_ev < 280) | (x_ev > 999))
    x_ev = x_ev[act_ind]
    y_ev = f_data["prophesee"]["left"]["y"][start_idx:end_idx][act_ind]
    p_ev = f_data["prophesee"]["left"]["p"][start_idx:end_idx][act_ind]
    t_ev = f_data["prophesee"]["left"]["t"][start_idx:end_idx][act_ind]

    return calc_exp(x_ev, y_ev, p_ev, t_ev, dist_map, filt_constants, exp_img, last_active, frame_interval, time_bin)


def convert_file(data_dir: str, seq_name: str, dt:float, output_time:float, alphas:np.ndarray,batch_size=10000) -> None:
    """
    Convert event data for a single sequence to exponential filter representation.
    
    Args:
        seq_name: Name of the sequence to process
    """
    print(f"Processing sequence: {seq_name}")
    
    # Configuration parameters
    time_bins = int(output_time / dt)    
    # File paths
    data_path = f"{data_dir}/events/m3ed/{seq_name}/{seq_name}_data.h5"
    output_path = f"{data_dir}/exp_filts/m3ed/{seq_name}.h5"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load event data
    try:
        f_data = h5py.File(data_path, "r")
    except Exception as e:
        print(f"Error loading data for {seq_name}: {e}")
        return

    # Setup camera parameters and filter constants
    dist_map, _ = load_camera_params(f_data)
    filt_constants = calc_filt_constants(alphas, time_bins)

    # Initialize exponential filter image
    exp_img = np.zeros((len(filt_constants), 360, 360), dtype=np.float32)
    
    # Create output file
    f_exp = h5py.File(output_path, "w")
    
    # Calculate total number of filter frames
    total_filts = int(
        (f_data["prophesee"]["left"]["t"][-1] - f_data["prophesee"]["left"]["t"][0]) # Times US
        * US_TO_S // (output_time*MS_TO_S)
    )
    
    last_active = f_data["prophesee"]["left"]["t"][0]

    # Create HDF5 datasets with compression
    exp_img_h5 = f_exp.create_dataset(
        "exp_filts",
        dtype=np.float32,
        shape=(total_filts, len(alphas), 360, 360),
        chunks=(1, len(alphas), 360, 360),
        **hdf5plugin.Blosc2(cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE),
    )

    exp_times_h5 = f_exp.create_dataset(
        "exp_times",
        dtype=np.float32,
        shape=(total_filts),
        chunks=(1000,),
    )

    # Process events in batches
    idx = 0
    ms_map = f_data["prophesee"]["left"]["ms_map_idx"][:]
    num_batches = len(ms_map) // batch_size
    
    start_time = time.time()
    for i in tqdm(range(num_batches), desc=f"Processing {seq_name}"):
        save_exp, exp_img, exp_times, last_active = process_event_batch(
            f_data, i, batch_size, dist_map, filt_constants, exp_img, last_active,frame_interval=output_time*MS_TO_S, time_bin=dt*MS_TO_S
        )
        
        if len(save_exp) > 0:
            exp_img_h5[idx:idx+len(save_exp)] = save_exp
            exp_times_h5[idx:idx+len(save_exp)] = exp_times
            idx += len(save_exp)

    processing_time = time.time() - start_time
    print(f"Sequence {seq_name} processed in {processing_time:.2f} seconds")
    
    # Resize datasets to actual size if needed
    if idx < total_filts:
        exp_img_h5.resize((idx, len(alphas), 360, 360))
        exp_times_h5.resize((idx,))
        
    f_exp.close()
    f_data.close()


def run(args: argparse.Namespace) -> None:
    """
    Process certain sequence file based on command-line arguments.
    
    Args:
        args: Command-line Sequence file arguments
    """
    m3ed_directory = args.data_dir
    
    try:
        files_ls = os.listdir(m3ed_directory+"/events/m3ed/")
    except FileNotFoundError:
        print(f"Error: Directory {m3ed_directory} not found.")
        sys.exit(1)
        
    if not files_ls:
        print("No matching sequence files found.")
        return
    
    files_ls.sort()

 
    try:
        convert_file(args.data_dir, files_ls[args.num].split('.h5')[0], args.dt, args.out_time, args.alphas)
    except Exception as e:
        print(f"Error processing file {files_ls[args.num]} ({args.num}): {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Inside Data Dir
    # {ds_name}/events/{seqs}  {ds_name}/exp_filts/{seqs}

    parser = argparse.ArgumentParser(
        description="Process event camera data to exponential filter representation."
    )
    parser.add_argument("--num", help="File number to process", type=int, default=0)
    parser.add_argument("--data_dir", help="Directory of M3ED Event Folders", type=str, default="/tmp/")
    parser.add_argument("--dt", help="Time Period of Filter Algo 2 \Delta T", type=float, default=0.2) # ms
    parser.add_argument("--out_time", help="Output Time, Slow Time Period Algo 2 \Delta R", type=float, default=7)# ms
    default_alphas = [0.12, 0.06, 0.03, 0.015, 0.0095, 0.0045]
    parser.add_argument("--alphas", help="Alpha values for the exponential filters (decay rates)",
                        type=float, nargs='+', default=default_alphas)
    args=parser.parse_args()
    run(args)
