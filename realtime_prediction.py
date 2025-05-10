import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

# Cargar modelo y etiquetas
model = tf.keras.models.load_model("gesture_model.h5")
GESTURES = ['pellizco', 'pulgar_afuera', 'puño_cerrado']  # cambia según tus clases
SEQUENCE_LENGTH = 30

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Búfer de secuencia
sequence = deque(maxlen=SEQUENCE_LENGTH)

# Webcam
cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Dibujar mano
    if results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]).flatten()
    else:
        keypoints = np.zeros(63)

    sequence.append(keypoints)

    # Solo predecir cuando hay 30 frames
    if len(sequence) == SEQUENCE_LENGTH:
        input_data = np.expand_dims(sequence, axis=0)  # shape: (1, 30, 63)
        prediction = model.predict(input_data, verbose=0)[0]
        predicted_label = GESTURES[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Mostrar en pantalla
        if confidence > 0.8:
            cv2.putText(image, f'{predicted_label} ({confidence:.2f})',
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow('Reconocimiento de Gestos', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
