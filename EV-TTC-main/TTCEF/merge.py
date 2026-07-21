import h5py
import numpy as np
import os
from tqdm import tqdm
import argparse
import hdf5plugin  # Required for LZ4 compression

# Train and test file lists
train_files = [
    "spot_indoor_stairs",
    "spot_forest_road_3",
    "spot_outdoor_night_penno_plaza_lights",
    "spot_outdoor_day_skatepark_2",
    "spot_outdoor_day_skatepark_1",
    "spot_outdoor_day_srt_under_bridge_2",
    "spot_outdoor_day_srt_green_loop",
    "spot_indoor_stairwell",
    "spot_outdoor_day_art_plaza_loop",
    "spot_outdoor_day_penno_short_loop",
    "car_forest_into_ponds_long",
    "car_urban_day_rittenhouse",
    "car_forest_into_ponds_short",
    "car_urban_night_rittenhouse",
    "car_urban_night_penno_big_loop",
    "car_urban_night_penno_small_loop_darker",
    "car_urban_night_city_hall",
    "car_urban_day_penno_big_loop",
    "car_urban_day_city_hall",
    "car_forest_tree_tunnel",
    "car_urban_day_ucity_small_loop",
]
test_files = [
    "car_urban_night_penno_small_loop",
    "car_urban_night_ucity_small_loop",
    "car_urban_day_penno_small_loop",
    "car_forest_sand_1",
    "spot_forest_easy_1",
    "spot_outdoor_day_srt_under_bridge_1",
    "spot_outdoor_day_rocky_steps",
    "spot_outdoor_night_penno_short_loop",
    "spot_forest_road_1",
    "spot_indoor_obstacles",
    "spot_indoor_building_loop",
]


def combine_files(data_dir, output_file, file_list, chunk_size=500):
    """
    Combine exponential filters and ground truth data into a single HDF5 file.

    Args:
        data_dir (str): Base directory containing the dataset.
        output_file (str): Path to the output HDF5 file.
        file_list (list): List of sequence names to process.
        chunk_size (int): Number of samples to process at a time.
    """
    with h5py.File(output_file, "w") as f_out:
        # Create datasets for combined data
        exp_dset = None
        ttc_dset = None
        flow_dset = None
        mask_dset = None
        indices_dset = f_out.create_dataset("indices", shape=(len(file_list), 2), dtype=np.int32)
        file_names_dset = f_out.create_dataset("file_names", shape=(len(file_list),), dtype=h5py.string_dtype())

        current_index = 0

        for i, seq_name in enumerate(tqdm(file_list, desc="Combining files")):
            exp_file = os.path.join(data_dir, "exp_filts/m3ed", f"{seq_name}.h5")
            gt_file = os.path.join(data_dir, "ttcef/m3ed", f"{seq_name}.h5")

            if not os.path.exists(exp_file) or not os.path.exists(gt_file):
                print(f"Skipping {seq_name}: Missing files.")
                continue

            # Load exponential filter data
            with h5py.File(exp_file, "r") as f_exp, h5py.File(gt_file, "r") as f_gt:
                exp_data = f_exp["exp_filts"]
                exp_times = f_exp["exp_times"]
                ttc_data = f_gt["ttc"]
                flow_data = f_gt["flow"]
                mask_data = f_gt["mask"]
                valid = f_gt["valid"]
                T = f_gt["T"]
                Omega = f_gt["Omega"]

                # Process data in chunks
                for start_idx in tqdm(range(0, exp_data.shape[0], chunk_size)):
                    end_idx = min(start_idx + chunk_size, exp_data.shape[0])

                    # Load chunk
                    exp_chunk = exp_data[start_idx:end_idx]
                    ttc_chunk = ttc_data[start_idx:end_idx]
                    flow_chunk = flow_data[start_idx:end_idx]
                    mask_chunk = mask_data[start_idx:end_idx]
                    valid_chunk = valid[start_idx:end_idx]
                    T_chunk = T[start_idx:end_idx]
                    Omega_chunk = Omega[start_idx:end_idx]

                    # Filter valid samples
                    if "car" in seq_name:  # Default filters for "car" sequences
                        valid_mask = valid_chunk & (np.linalg.norm(T_chunk, axis=1) > 1.3) & (np.linalg.norm(Omega_chunk, axis=1) < 0.18)
                    else:  # Default filters for other sequences
                        valid_mask = valid_chunk & (np.linalg.norm(T_chunk, axis=1) > 0.25) & (np.linalg.norm(Omega_chunk, axis=1) < 0.18)

                    exp_chunk = exp_chunk[valid_mask]
                    ttc_chunk = ttc_chunk[valid_mask]
                    flow_chunk = flow_chunk[valid_mask]

                    mask_chunk = mask_chunk[valid_mask] & (np.abs(exp_chunk[:, -1]) > 1e-3) & ~np.isnan(ttc_chunk)  # Exp filter mask
                    ttc_chunk = np.nan_to_num(ttc_chunk)
                    mask_chunk = mask_chunk & (ttc_chunk < 100)# Maximum TTC Mask

                    # Skip if no valid samples
                    if exp_chunk.shape[0] == 0:
                        continue

                    # Initialize datasets if not already created
                    if exp_dset is None:
                        exp_dset = f_out.create_dataset(
                            "exp_filts",
                            shape=(0, *exp_chunk.shape[1:]),
                            maxshape=(None, *exp_chunk.shape[1:]),
                            dtype=np.float16,
                            chunks=(1, *exp_chunk.shape[1:]),
                           **hdf5plugin.Blosc2(cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE)
                        )
                        ttc_dset = f_out.create_dataset(
                            "ttc",
                            shape=(0, *ttc_chunk.shape[1:]),
                            maxshape=(None, *ttc_chunk.shape[1:]),
                            dtype=np.float32,
                            chunks=(1, *ttc_chunk.shape[1:]),
                           **hdf5plugin.Blosc2(cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE)
                        )
                        flow_dset = f_out.create_dataset(
                            "flow",
                            shape=(0, *flow_chunk.shape[1:]),
                            maxshape=(None, *flow_chunk.shape[1:]),
                            dtype=np.float16,
                            chunks=(1, *flow_chunk.shape[1:]),
                            **hdf5plugin.Blosc2(cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE)
                        )
                        mask_dset = f_out.create_dataset(
                            "mask",
                            shape=(0, *mask_chunk.shape[1:]),
                            maxshape=(None, *mask_chunk.shape[1:]),
                            dtype=bool,
                            chunks=(1, *mask_chunk.shape[1:]),
                            **hdf5plugin.Blosc2(cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE)
                        )

                    # Append valid data to datasets
                    new_size = current_index + exp_chunk.shape[0]
                    exp_dset.resize((new_size, *exp_chunk.shape[1:]))
                    ttc_dset.resize((new_size, *ttc_chunk.shape[1:]))
                    flow_dset.resize((new_size, *flow_chunk.shape[1:]))
                    mask_dset.resize((new_size, *mask_chunk.shape[1:]))

                    exp_dset[current_index:new_size] = exp_chunk
                    ttc_dset[current_index:new_size] = ttc_chunk
                    flow_dset[current_index:new_size] = flow_chunk
                    mask_dset[current_index:new_size] = mask_chunk

                    current_index = new_size

            # Record indices and file name
            indices_dset[i] = [current_index, new_size]
            file_names_dset[i] = seq_name

        print(f"Combined data saved to {output_file}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Combine exponential filters and ground truth data into a single HDF5 file.")
    parser.add_argument("--data_dir", type=str, required=True, help="Base directory containing the dataset.")
    parser.add_argument("--out_dir", type=str, required=True, help="Base directory containing the dataset.")
    args = parser.parse_args()

    # Base directory containing the dataset
    DATA_DIR = args.data_dir

    # Output files
    TRAIN_OUTPUT = os.path.join(args.out_dir, "train.h5")
    TEST_OUTPUT = os.path.join(args.out_dir, "test.h5")

    # Combine train and test files
    combine_files(DATA_DIR, TRAIN_OUTPUT, train_files)
    combine_files(DATA_DIR, TEST_OUTPUT, test_files)
