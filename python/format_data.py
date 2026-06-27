import numpy as np
import os
from paths import DATASETS_DIR
from variables import WINDOW_SIZE, OVERLAP


def apply_sliding_window(data, window_size, overlap):
    height, width = np.int32(
        np.floor((data.shape[0] - window_size)/overlap)) + 1, window_size
    data_format = np.zeros(shape=(height, width))
    for i in range(height):
        start_idx = np.int32(i*overlap)
        end_idx = start_idx + window_size
        data_format[i, :] = data[start_idx:end_idx, 0]
    return data_format


if __name__ == "__main__":
    dataset_path = str(DATASETS_DIR)
    normal_files = os.path.join(dataset_path, "format", "normal_data.npy")
    normal_data = np.load(normal_files)
    # formatting data, data sample = 48KHz, 1024 sample ~ 21 ms, 128 sample ~ 5 ms
    print("formatting data...")
    window_size = WINDOW_SIZE
    overlap = OVERLAP
    data_format = apply_sliding_window(normal_data, window_size, overlap)
    print("Data formatted shape: ", data_format.shape)
    # save formatted data
    save_path = os.path.join(dataset_path, "format", "normal_data_overlap.npy")
    np.save(save_path, data_format)
