import matplotlib.pyplot as plt
import os
import numpy as np
from paths import DATASETS_DIR

if __name__ == "__main__":
    dataset_path = str(DATASETS_DIR)
    # load the normal data
    normal_files = os.path.join(dataset_path, "format", "normal_data.npy")
    normal_data = np.load(normal_files)

    # load the ball defect data
    ball_defect_files = os.path.join(
        dataset_path, "format", "ball_defect_data.npy")
    ball_defect_data = np.load(ball_defect_files)

    # load the inner defect data
    inner_defect_files = os.path.join(
        dataset_path, "format", "inner_defect_data.npy")
    inner_defect_data = np.load(inner_defect_files)

    # load the outer defect data
    outer_defect_files = os.path.join(
        dataset_path, "format", "outer_defect_data.npy")
    outer_defect_data = np.load(outer_defect_files)

    # clip the data to the length of the normal data
    min_length = normal_data.shape[0]
    ball_defect_data = ball_defect_data[:min_length]
    inner_defect_data = inner_defect_data[:min_length]
    outer_defect_data = outer_defect_data[:min_length]

    # load the features
    features_files = os.path.join(dataset_path, "features", "features.npy")
    features = np.load(features_files)
    # print data shape
    print("Data shape: ", normal_data.shape)
    print("Features shape: ", features.shape)
    # plot data and keep the figure visible until the window is closed
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(normal_data, color='green')
    plt.title('Normal Data')
    plt.subplot(4, 1, 2)
    plt.plot(ball_defect_data, color='red')
    plt.title('Ball Defect Data')
    plt.subplot(4, 1, 3)
    plt.plot(inner_defect_data, color='blue')
    plt.title('Inner Defect Data')
    plt.subplot(4, 1, 4)
    plt.plot(outer_defect_data, color='orange')
    plt.title('Outer Defect Data')
    plt.tight_layout()
    plt.show()
