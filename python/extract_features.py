import numpy as np
import os
from paths import DATASETS_DIR


def extract_features(data):
    # compute mean
    mean = np.mean(data, axis=1).reshape(-1, 1)
    # computes standard deviation
    std = np.std(data, axis=1).reshape(-1, 1)
    # variance
    var = np.var(data, axis=1).reshape(-1, 1)
    # compute root mean square
    rms = np.sqrt(np.mean(np.square(data), axis=1)).reshape(-1, 1)
    # mean absolute value
    mav = np.mean(np.abs(data), axis=1).reshape(-1, 1)
    # Compute an intermediate data that X - mean
    data_minus_mean = data - mean
    # compute Kurtosis
    ku = (np.mean(np.power(data_minus_mean, 4), axis=1).reshape(-1, 1) /
          np.power(std, 4)).reshape(-1, 1)
    # max abs
    max_abs = np.max(np.abs(data), axis=1).reshape(-1, 1)
    # crest factor
    cf = (max_abs / rms).reshape(-1, 1)
    # impulse factor
    impf = (max_abs / mav).reshape(-1, 1)
    # peak to peak amplitude
    p2p = (np.max(data, axis=1) - np.min(data, axis=1)).reshape(-1, 1)
    # skewness
    sw = (np.mean(np.power(data_minus_mean, 3), axis=1).reshape(-1, 1) /
          np.power(std, 3).reshape(-1, 1)).reshape(-1, 1)
    # Energy
    en = (np.square(np.mean(np.sqrt(np.abs(data)), axis=1))).reshape(-1, 1)
    features = np.concatenate(
        (mean, var, rms, ku, cf, impf, p2p, sw, en), axis=1)
    return features


if __name__ == "__main__":
    dataset_path = str(DATASETS_DIR)
    data = np.load(os.path.join(
        dataset_path, "format", "normal_data_overlap.npy"))
    features = extract_features(data)
    print("Featurs shape: ", features.shape)
    # save the features
    save_path = os.path.join(dataset_path, "features")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, "features.npy"), features)
    print("features saved into ", save_path)
