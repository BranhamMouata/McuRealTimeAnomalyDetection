import os
from preprocess_features import preprocess
from model import build_autoencoder, train_autoencoder, reconstruction_errors, choose_threshold
from paths import PROJECT_ROOT
import matplotlib.pyplot as plt
import joblib

if __name__ == "__main__":
    # load the data
    (X_train, X_val, X_test, scaler) = preprocess()
    # create the model
    autoencoder, encoder, decoder = build_autoencoder(input_dim=X_train.shape[1],
                                                      latent_dim=4, hidden_dims=[32, 16])
    autoencoder.summary()
    # train the model
    model_save_path = os.path.join(str(PROJECT_ROOT), "model")
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    history = train_autoencoder(autoencoder, X_train, X_val, epochs=100, batch_size=32,
                                save_path=os.path.join(model_save_path, "cwru_autoencoder.h5"), verbose=1)
    # evaluate the model
    loss = autoencoder.evaluate(X_test, X_test)
    print(f"Test Loss: {loss}")
    # show the training history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(model_save_path, "loss_plot.png"))
    # Compute the threshold for anomaly detection based on the training loss
    train_reconstruction_errors = reconstruction_errors(autoencoder, X_train)
    threshold = choose_threshold(train_reconstruction_errors)
    # save the threshold
    joblib.dump(threshold, os.path.join(
        model_save_path, "error_threshold.pkl"))
