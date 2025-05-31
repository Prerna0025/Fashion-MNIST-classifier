import matplotlib.pyplot as plt

class_names_default = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def plot_sample_image(X, y):
    """Display a single sample image from the dataset."""
    plt.imshow(X[0], cmap='binary')
    plt.axis('off')
    plt.title(f"Label: {y[0]}")
    plt.show()

def plot_image_grid(X, y, class_names=class_names_default, n_rows=4, n_cols=10):
    """Display a grid of sample images with their class names."""
    plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(X[index], cmap='binary', interpolation='nearest')
            plt.axis('off')
            plt.title(class_names[y[index]], fontsize=9)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.show()
