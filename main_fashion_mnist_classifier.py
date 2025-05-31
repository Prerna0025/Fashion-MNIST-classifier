import numpy as np
from utils.visualization import plot_sample_image, plot_image_grid
from data.loader import load_and_preprocess_data
from model.architecture import build_model
from model.training import compile_and_train_model, plot_training_curves, evaluate_and_predict

if __name__ == '__main__':
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_and_preprocess_data()

    plot_sample_image(X_train, y_train)
    plot_image_grid(X_train, y_train, class_names)

    model = build_model()
    history = compile_and_train_model(model, X_train, y_train, X_valid, y_valid)

    plot_training_curves(history)
    evaluate_and_predict(model, X_test, y_test)