import kagglehub
import os
import scipy.io as sio
import numpy as np
from paths import DATASETS_DIR


def extract_data(files, dataset_path):
    data = []
    for file in files:
        file_path = os.path.join(dataset_path, "raw", file)
        # Only use the DE (Drive End)time
        key = file.split(".")[0]
        key = "X"+key[len(key)-3:len(key)] + "_DE_time"
        mat_data = sio.loadmat(file_path)[key].flatten().tolist()
        data.extend(mat_data)
    return np.array(data, dtype=np.float32).reshape(-1, 1)


if __name__ == "__main__":
    dataset_path = str(DATASETS_DIR)
    # create datasets directory if it doesn't exist
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Download the dataset
    print("Downloading dataset ...")
    path = kagglehub.dataset_download(
        "brjapon/cwru-bearing-datasets", output_dir=dataset_path)
    print(f"Dataset downloaded to: {path}")
    # Extract data
    print("Extracting data...")
    ball_defect_files = ["B007_1_123.mat", "B014_1_190.mat", "B021_1_227.mat"]
    inner_defect_files = ["IR007_1_110.mat",
                          "IR014_1_175.mat", "IR021_1_214.mat"]
    outer_defect_files = ["OR007_6_1_136.mat",
                          "OR014_6_1_202.mat", "OR021_6_1_239.mat"]
    normal_files = ["Time_Normal_1_098.mat"]

    ball_defect_data = extract_data(ball_defect_files, dataset_path)
    inner_defect_data = extract_data(inner_defect_files, dataset_path)
    outer_defect_data = extract_data(outer_defect_files, dataset_path)
    normal_data = extract_data(normal_files, dataset_path)

    print(f"Ball defect data size: {ball_defect_data.size}")
    print(f"Inner defect data size: {inner_defect_data.size}")
    print(f"Outer defect data size: {outer_defect_data.size}")
    print(f"Normal data size: {normal_data.size}")

    # save data
    save_path = os.path.join(dataset_path, "format")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, "normal_data.npy"), normal_data)
    np.save(os.path.join(save_path, "ball_defect_data.npy"), ball_defect_data)
    np.save(os.path.join(save_path, "inner_defect_data.npy"), inner_defect_data)
    np.save(os.path.join(save_path, "outer_defect_data.npy"), outer_defect_data)
    print(f"data saved into {save_path}")
