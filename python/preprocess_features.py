from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from paths import DATASETS_DIR
import numpy as np
import os
import joblib


def preprocess():
    # load the data
    dataset_path = str(DATASETS_DIR)
    data = np.load(os.path.join(dataset_path, "features", "features.npy"))
    # split the data into train, validation, and test setsS
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=10)
    # save the train and test sets
    save_path = os.path.join(dataset_path, "train_test_split")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, "X_train.npy"), X_train)
    np.save(os.path.join(save_path, "X_test.npy"), X_test)
    # standardize the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    # save the scaler
    joblib.dump(scaler, os.path.join(save_path, "scaler.pkl"))
    # apply the scaler to the datasets
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return (X_train, X_val, X_test, scaler)
