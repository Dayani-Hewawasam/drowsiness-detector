import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import pygame
import time
import csv
from datetime import datetime

# ---------------- CONFIG ----------------
EAR_THRESHOLD = 0.25         # Below this = eyes closed
EAR_CONSEC_FRAMES = 20       # Frames closed before triggering alarm
MAR_THRESHOLD = 0.6          # Above this = yawning
ALARM_SOUND_PATH = "alarm.wav"
LOG_FILE = "drowsiness_log.csv"

# ---------------- SETUP ----------------
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound(ALARM_SOUND_PATH)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices for MediaPipe Face Mesh (468 points)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]

# ---------------- FUNCTIONS ----------------
def eye_aspect_ratio(landmarks, eye_points, w, h):
    coords = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye_points]
    A = dist.euclidean(coords[1], coords[5])
    B = dist.euclidean(coords[2], coords[4])
    C = dist.euclidean(coords[0], coords[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(landmarks, mouth_points, w, h):
    coords = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in mouth_points]
    A = dist.euclidean(coords[1], coords[7])
    B = dist.euclidean(coords[2], coords[6])
    C = dist.euclidean(coords[3], coords[5])
    D = dist.euclidean(coords[0], coords[4])
    mar = (A + B + C) / (3.0 * D)
    return mar

def log_event(event_type):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event_type])

# Create log file with header if not exists
try:
    with open(LOG_FILE, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "event"])
except FileExistsError:
    pass

# ---------------- MAIN LOOP ----------------
cap = cv2.VideoCapture(0)
counter = 0
alarm_on = False

print("Starting drowsiness detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    status_text = "No face detected"
    status_color = (0, 0, 255)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(landmarks, MOUTH, w, h)

        # Draw eye/mouth points for visual confirmation
        for idx in LEFT_EYE + RIGHT_EYE + MOUTH:
            point = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            cv2.circle(frame, point, 2, (0, 255, 0), -1)

        if ear < EAR_THRESHOLD:
            counter += 1
            if counter >= EAR_CONSEC_FRAMES:
                status_text = "DROWSINESS ALERT!"
                status_color = (0, 0, 255)
                if not alarm_on:
                    alarm_on = True
                    alarm_sound.play(loops=-1)
                    log_event("drowsiness_detected")
        else:
            counter = 0
            if alarm_on:
                alarm_sound.stop()
                alarm_on = False
            status_text = "Active"
            status_color = (0, 255, 0)

        if mar > MAR_THRESHOLD:
            cv2.putText(frame, "YAWNING", (w - 180, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            log_event("yawn_detected")

        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        if alarm_on:
            alarm_sound.stop()
            alarm_on = False

    cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    cv2.imshow("Drowsiness Detection - Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Session ended. Log saved to", LOG_FILE)