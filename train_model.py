import numpy as np
import tensorflow as tf
import pandas as pd
import glob

SEQUENCE_LENGTH = 30
NUM_KEYPOINTS = 63  # 21 puntos * (x, y, z)
NUM_CLASSES = 3     # Ajusta esto según tu lista de gestos

def load_data(path='dataset/keypoints_csv/*.csv'):
    files = glob.glob(path)
    sequences = []
    labels = []
    for f in files:
        data = pd.read_csv(f, header=None).values
        for row in data:
            sequences.append(row[:-1].reshape(SEQUENCE_LENGTH, NUM_KEYPOINTS))
            labels.append(int(row[-1]))
    return np.array(sequences), tf.keras.utils.to_categorical(labels, NUM_CLASSES)

X, y = load_data()

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, NUM_KEYPOINTS)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2)

model.save("gesture_model.h5")
