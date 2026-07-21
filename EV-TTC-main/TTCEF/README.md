<!-- filepath: /home/abisulco/EV-TTC/TTCEF/README.md -->
### $T^2CEF$ Optical Flow and TTC Ground Truth
This folder contains scripts to process event camera data from the M3ED dataset and generate exponential filter representations, time-to-contact (TTC), and optical flow ground truth. The generated data can be merged into train/test datasets for machine learning tasks.

## Folder Layout

### Scripts

#### `create_exp.py`
- Generates exponential filter representations from event camera data
- Mirrors the real-time C++ implementation for processing events
- Outputs HDF5 files containing exponential filter images and timestamps

#### `calc_gt.py`
- Calculates ground truth TTC, optical flow, and depth using camera poses and depth maps
- Handles camera calibration, pose interpolation, and depth projection
- Outputs HDF5 files with compressed datasets for TTC, flow, and masks

#### `merge.py`
- Combines multiple HDF5 files (exponential filters and ground truth) into train and test datasets
- Filters invalid samples and applies compression for efficient storage
- Outputs merged HDF5 files with metadata for indexing

#### `down.py`
- Generates a script (`down.sh`) to download the M3ED dataset from S3 storage
- Uses the Amazon S3 client for faster downloads

#### `submit.sh`
- SLURM batch script to run `create_exp.py` and `calc_gt.py` in parallel for multiple sequences
- Configures job parameters like memory, CPUs, and excluded nodes

### Output Structure

The scripts generate the following outputs:

#### Exponential Filters (`exp_filts`)
- Directory: `/exp_filts/m3ed/`
- File format: `<sequence_name>.h5`
- Contains:
  - `exp_filts`: Exponential filter images
  - `exp_times`: Timestamps for each filter frame

#### Ground Truth (`ttcef`)
- Directory: `/ttcef/m3ed/`
- File format: `<sequence_name>.h5`
- Contains:
  - `ttc`: Time-to-contact ground truth
  - `flow`: Optical flow ground truth
  - `depth`: Depth maps
  - `mask`: Validity masks
  - Metadata: Camera intrinsics, poses, and timestamps

#### Merged Datasets
- Directory: / (specified in `merge.py`)
- File format: `train.h5`, `test.h5`
- Contains:
  - Combined exponential filters, TTC, flow, and masks
  - Metadata: Indices and sequence names

## How to Use

### 1. Download M3ED Dataset
Run `down.py` to generate a script for downloading the M3ED dataset.

Example Command:
```
python down.py
chmod a+x down.sh
./down.sh
```

### 2. Generate Exponential Filters
Run `create_exp.py` to process event camera data and generate exponential filter representations.

Example Command:
```
python create_exp.py --data_dir /path/to/dataset --num 0
```

Arguments:
- `--data_dir`: Path to the dataset directory
- `--num`: Sequence index to process
- `--dt`: Time bin resolution for filters (default: 0.2 ms)
- `--out_time`: Output time for filters (default: 7 ms)
- `--alphas`: Alpha values for exponential decay rates (default: [0.12, 0.06, 0.03, 0.015, 0.0095, 0.0045])

### 3. Calculate Ground Truth
Run `calc_gt.py` to calculate TTC, optical flow, and depth ground truth.

Example Command:
```
python calc_gt.py --data_dir /path/to/dataset --num 0
```

Arguments:
- `--data_dir`: Path to the dataset directory
- `--num`: Sequence index to process
- `--dt`: Time interval for optical flow calculation (default: 0.2 ms)
- `--splat`: Number of pixels to splat for depth points (default: 3)
- `--full_res`: Use full resolution (720x1280) instead of cropped 360x360

### 4. Merge Files
Run `merge.py` to combine exponential filters and ground truth into train/test datasets.

Example Command:
```
python merge.py --data_dir /path/to/dataset --out_dir /path/to/output
```

Arguments:
- `--data_dir`: Path to the dataset directory
- `--out_dir`: Path to save the merged train/test datasets



### (Run on SLURM Cluster)
Use `submit.sh` to run `create_exp.py` and `calc_gt.py` in parallel for multiple sequences.

Example Command:
```
sbatch submit.sh
```

### Caveats
The code for generating optical flow and TTC ground truth has been optimized for speed, incorporating several adjustments. Instead of using hidden point removal (HPR) as in the original M3ED repository, we employ splatting, which is computationally faster. Additionally, we do not use the global depth maps provided by M3ED, as they can result in duplicate objects across multiple loops due to the absence of loop closure. As a simpler alternative, we use the closest depth map to the current pose, transform the depth into the frame and splat it. More advanced approaches, such as combining multiple depth frames and transforming them into the current frame, could also be applied. Furthermore, SE(3) interpolation is used between pose intervals, which may introduce inaccuracies during non-smooth motion. Lastly, only ego-flow is calculated, meaning optical flow caused by independently moving objects is not included.
