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


def model_performance(normal_loss, defect_loss):
    gt_defect_labels = np.concatenate((np.zeros(
        (normal_features.shape[0])), np.ones((normal_features.shape[0]))), axis=0).flatten()

    defect_detection = (np.concatenate(
        (normal_loss, defect_loss), axis=0) >= threshold).astype(np.int8).flatten()
    # Accuracy
    accuracy = np.mean((defect_detection ==
                       gt_defect_labels).astype(np.int8))
    # Confusion parameters
    tp = sum((defect_detection == 1) & (gt_defect_labels == 1))
    fp = sum((defect_detection == 1) & (gt_defect_labels == 0))
    fn = sum((defect_detection == 0) & (gt_defect_labels == 1))
    # Recall
    recall = tp/(tp + fn)
    # Precision
    precision = tp/(tp + fp)
    return accuracy, recall, precision


def evaluate_keras_model(model_path, normal_features, defect_features, threshold):
    # Load the Keras model
    model = load_model(model_path)
    # evaluate the model on the normal features
    normal_loss = reconstruction_errors(model, normal_features)
    # evaluate the model on the defect features
    defect_loss = reconstruction_errors(model, defect_features)

    # Model performance
    return model_performance(normal_loss, defect_loss)


def litert_infer(input_data, interpreter, input_details, output_details, input_scale,
                 input_zero_point, output_scale, output_zero_point):
    quantized = np.rint(input_data / input_scale + input_zero_point)
    quantized = np.clip(quantized, -128, 127).astype(np.int8)
    input_data = np.expand_dims(
        quantized, axis=0).astype(input_details["dtype"])
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
    # Model performance
    return model_performance(normal_loss, defect_loss)


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

    # evaluate the keras model
    k_accuracy, k_recall, k_precision = evaluate_keras_model(
        keras_model_path, normal_features, defect_features, threshold)
    # evaluate the litert model
    litert_model_path = os.path.join(
        str(PROJECT_ROOT), "model", "cwru_autoencoder_quant8.tflite")
    lt_accuracy, lt_recall, lt_precision = evaluate_litert_model(
        litert_model_path, normal_features, defect_features, threshold)
    print("Keras model evaluation: ")
    print(" Keras model [accuracy, recall, precision] : [{}, {}, {}]".format(
          k_accuracy, k_recall, k_precision))
    print("LiteRT model evaluation: ")
    print(" LiteRT model [accuracy, recall, precision] : [{}, {}, {}]".format(
          lt_accuracy, lt_recall, lt_precision))
