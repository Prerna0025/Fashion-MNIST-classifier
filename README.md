# Fashion MNIST Classifier

A TensorFlow-based deep learning project to classify images from the Fashion MNIST dataset. The project is built with modular and readable code, organized into separate components for data loading, model architecture, training, visualization, and evaluation.

---

## Project Goal

This project aims to:

* Build a multi-layer neural network to classify fashion images.
* Provide a clean and professional codebase for learners and practitioners.
* Highlight the importance of model organization and clarity.

---

## Features

* Dataset preprocessing and normalization
* Neural network with:

  * Dense layers
  * ReLU activations
  * Softmax output
  * L2 regularization
* Training with validation
* Evaluation on test data
* Visualizations of:

  * Training samples
  * Image grids
  * Training and validation accuracy/loss curves

---

## Dataset

The project uses the **Fashion MNIST** dataset from Keras:

* 60,000 training images
* 10,000 test images
* 10 clothing categories:

  * T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

Each image is a 28x28 grayscale image labeled with one of the 10 classes.

---

## Folder Structure

```
project-root/
├── fashion_mnist_classifier.py      # Main entry script
├── data/
│   └── loader.py                    # Handles dataset loading and preprocessing
├── model/
│   ├── architecture.py              # Model definition
│   └── training.py                  # Training, evaluation, plotting logic
├── utils/
│   └── visualization.py             # Image plotting utilities
```

---

## Requirements

* Python 3.7+
* TensorFlow
* NumPy
* Pandas
* Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

To run the training script:

```bash
python fashion_mnist_classifier.py
```

---

## Output

* A grid of sample training images with labels
* Training and validation accuracy/loss curves
* Final test accuracy and predictions

---

## License

This project is open source and available under the MIT License.

---

Feel free to fork, modify, or extend this project for your learning or development needs!
