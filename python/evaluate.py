import os
from paths import DATASETS_DIR, PROJECT_ROOT
import numpy as np
import joblib
from variables import WINDOW_SIZE, OVERLAP
from format_data import apply_sliding_window
from extract_features import extract_features
from tensorflow.keras.models import load_model
from model import reconstruction_errors
import tensorflow as tf


def evaluate_keras_model(model_path, normal_features, defect_features, threshold):
    # Load the Keras model
    model = load_model(model_path)
    # evaluate the model on the normal features
    normal_loss = reconstruction_errors(model, normal_features)
    # evaluate the model on the defect features
    defect_loss = reconstruction_errors(model, defect_features)

    # classify the normal features
    normal_predictions = (normal_loss < threshold).astype(int)
    # classify the defect features
    defect_predictions = (defect_loss >= threshold).astype(int)

    # compute the accuracy of the normal features
    normal_accuracy = np.mean(normal_predictions)
    # compute the accuracy of the defect features
    defect_accuracy = np.mean(defect_predictions)
    return normal_accuracy, defect_accuracy


def litert_infer(input_data, interpreter, input_details, output_details, input_scale,
                 input_zero_point, output_scale, output_zero_point):
    input_data = input_data / input_scale + input_zero_point
    input_data = np.expand_dims(
        input_data, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details["index"])[0]
    output_data = (output_data.astype(np.float32) -
                   output_zero_point) * output_scale
    return output_data


def evaluate_litert_model(model_path, normal_features, defect_features, threshold):
    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    if (input_details["dtype"] != np.int8):
        raise ValueError(
            "The input type of the TFLite model is not int8. Please check the model.")
    if (output_details["dtype"] != np.int8):
        raise ValueError(
            "The output type of the TFLite model is not int8. Please check the model.")
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]
    normal_loss = np.zeros(normal_features.shape[0])
    defect_loss = np.zeros(defect_features.shape[0])
    for idx in range(normal_features.shape[0]):
        # normal data
        input_data = normal_features[idx, :].astype(np.float32)
        output_data = litert_infer(input_data, interpreter, input_details, output_details, input_scale,
                                   input_zero_point, output_scale, output_zero_point)
        normal_loss[idx] = np.mean(
            np.square(input_data.astype(np.float32) - output_data))
        # defect data
        input_data = defect_features[idx, :].astype(np.float32)
        output_data = litert_infer(input_data, interpreter, input_details, output_details, input_scale,
                                   input_zero_point, output_scale, output_zero_point)
        defect_loss[idx] = np.mean(
            np.square(input_data.astype(np.float32) - output_data))
    normal_accuracy = np.mean(normal_loss < threshold)
    defect_accuracy = np.mean(defect_loss >= threshold)
    return normal_accuracy, defect_accuracy


if __name__ == "__main__":
    # load normal features test
    normal_features = np.load(os.path.join(
        DATASETS_DIR, "train_test_split", "X_test.npy"))
    # load defect data
    defect = np.load(os.path.join(
        DATASETS_DIR, "format", "ball_defect_data.npy"))
    defect = np.concatenate((defect, np.load(os.path.join(
        DATASETS_DIR, "format", "inner_defect_data.npy"))))
    defect = np.concatenate((defect, np.load(os.path.join(
        DATASETS_DIR, "format", "outer_defect_data.npy"))))
    # shuffle the defect data
    rng = np.random.default_rng(seed=42)
    rng.shuffle(defect)
    # apply sliding window to the defect data
    defect = apply_sliding_window(defect, WINDOW_SIZE, OVERLAP)
    # truncate the defect data to match the normal data size
    defect = defect[:normal_features.shape[0], :]
    # extract features from the defect data
    defect_features = extract_features(defect)
    print("Data shape: ", normal_features.shape)
    print("Defect features shape: ", defect_features.shape)
    # load the scaler
    scaler = joblib.load(os.path.join(
        DATASETS_DIR, "train_test_split", "scaler.pkl"))
    # scale the normal features
    normal_features = scaler.transform(normal_features)
    # scale the defect features
    defect_features = scaler.transform(defect_features)

    # load the model
    keras_model_path = os.path.join(
        str(PROJECT_ROOT), "model", "cwru_autoencoder.h5")
    # load the threshold
    threshold = joblib.load(os.path.join(
        str(PROJECT_ROOT), "model", "error_threshold.pkl"))

    # evaluate the keras model on the normal features
    keras_normal_accuracy, keras_defect_accuracy = evaluate_keras_model(
        keras_model_path, normal_features, defect_features, threshold)
    # evaluate the litert model on the normal features
    litert_model_path = os.path.join(
        str(PROJECT_ROOT), "model", "cwru_autoencoder_quant8.tflite")
    litert_normal_accuracy, litert_defect_accuracy = evaluate_litert_model(
        litert_model_path, normal_features, defect_features, threshold)
    print("Keras model evaluation: ")
    print("Normal accuracy: ", keras_normal_accuracy)
    print("Defect accuracy: ", keras_defect_accuracy)
    print("LiteRT model evaluation: ")
    print("Normal accuracy: ", litert_normal_accuracy)
    print("Defect accuracy: ", litert_defect_accuracy)
