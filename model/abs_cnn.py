import tensorflow as tf
from model.cbam import CBAM

def build_abs_cnn(input_shape=(28, 28, 2), num_classes=10):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, activation='relu', padding='same')(inputs)
    x = CBAM()(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, activation='relu', padding='same')(x)
    x = CBAM()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1600, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
