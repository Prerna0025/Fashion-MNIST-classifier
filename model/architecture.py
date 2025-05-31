from tensorflow import keras

def build_model():
    return keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dense(100, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dense(10, activation='softmax')
    ])