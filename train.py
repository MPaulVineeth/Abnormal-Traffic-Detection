import os
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from model.abs_cnn import build_abs_cnn

def load_data(folder="data_preprocessing/dataset/images"):
    x, y = [], []
    class_names = set()

    files = [f for f in os.listdir(folder) if f.endswith(".npy")]

    for fname in files:
        img = np.load(os.path.join(folder, fname))
        label_name = fname.split("_")[0]
        class_names.add(label_name)
        x.append(img)
        y.append(label_name)

    class_names = sorted(list(class_names))
    label_map = {name: idx for idx, name in enumerate(class_names)}
    y = [label_map[label] for label in y]

    x = np.array(x) / 255.0
    x = np.expand_dims(x, axis=-1)
    x = np.concatenate([x, x], axis=-1)

    y = tf.keras.utils.to_categorical(y, num_classes=len(class_names))
    return train_test_split(x, y, test_size=0.2), class_names

def main():
    (x_train, x_test, y_train, y_test), class_names = load_data()
    model = build_abs_cnn(input_shape=(28, 28, 2), num_classes=len(class_names))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=20, batch_size=20, validation_split=0.1)

    # ✅ Save training history
    with open("history.pkl", "wb") as f:
        pickle.dump(history.history, f)

    model.save("abs_cnn_model.h5")
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✅ Final ABS-CNN Accuracy: {acc * 100:.2f}% | Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
