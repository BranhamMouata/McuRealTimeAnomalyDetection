from paths import PROJECT_ROOT, DATASETS_DIR, MODEL_DIR
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import joblib


def convert(features, model, output_path):
    def representative_dataset():
        for data in tf.data.Dataset.from_tensor_slices((features)).batch(1).take(500):
            yield [tf.dtypes.cast(data, tf.float32)]

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_quant_model = converter.convert()

    # Save the model.
    with open(output_path, 'wb') as f:
        f.write(tflite_quant_model)
    return tflite_quant_model


if __name__ == "__main__":
    # Load the dataset for representative dataset
    features = np.load(DATASETS_DIR / "features" / "features.npy")
    # load scaler
    scaler = joblib.load(DATASETS_DIR / "train_test_split" / "scaler.pkl")
    # scale the features
    features = scaler.transform(features)
    # shuffle the data
    rng = np.random.default_rng(seed=42)
    rng.shuffle(features)
    # load keras model
    keras_model = load_model(MODEL_DIR / "cwru_autoencoder.h5")
    output_path = MODEL_DIR / "cwru_autoencoder_quant8.tflite"
    tflite_quant_model = convert(features, keras_model, output_path)
