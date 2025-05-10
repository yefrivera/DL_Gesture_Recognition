import cv2
import mediapipe as mp
import numpy as np
import os
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = 'dataset/keypoints_csv'
GESTURES = ['pellizco', 'pulgar_afuera', 'puño_cerrado']  # Ejemplo

SEQUENCE_LENGTH = 30

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]).flatten()
    else:
        return np.zeros(21 * 3)

def process_video(video_path, label, output_file):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        if len(sequence) == SEQUENCE_LENGTH:
            with open(output_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(np.concatenate(sequence + [np.array([label])]))
            sequence.pop(0)
    cap.release()

# Ejemplo: procesar todos los videos de un gesto
for gesture in GESTURES:
    gesture_folder = f'dataset/videos_for_training/{gesture}'
    for video in os.listdir(gesture_folder):
        process_video(os.path.join(gesture_folder, video), GESTURES.index(gesture),
                      f'{DATA_PATH}/{gesture}.csv')
