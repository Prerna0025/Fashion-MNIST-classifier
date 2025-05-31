import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

def compile_and_train_model(model, X_train, y_train, X_valid, y_valid):
    tf.keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

def plot_training_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.title("Training and Validation Metrics")
    plt.show()

def evaluate_and_predict(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
    X_new = X_test[:3]
    y_pred = np.argmax(model.predict(X_new), axis=1)
    print("\nPredictions for first 3 samples:", y_pred)

