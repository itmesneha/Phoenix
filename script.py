import pickle
import gzip
import os
import cv2
import mediapipe as mp
import numpy as np


with gzip.open('phoenix14t.pami0.dev.annotations_only.gzip', 'rb') as f:
    annotations = pickle.load(f)

print('type: ', type(annotations))
sample = annotations[0]
print(type(sample))
print(sample.keys())
print('name: ', sample['name'])
print('gloss: ', sample['gloss'])

video_path = f"videos/{sample['name']}.mp4"
print(os.path.exists(video_path))

# 1️⃣ Extract pose from one video using MediaPipe
# 2️⃣ Define final pose tensor format + storage
# 3️⃣ Build dataset class (gloss + pose)
# 4️⃣ Visualize pose as a stick figure (very important sanity check)

# 👉 I strongly recommend 1 → 4 → 3 → 2, but your call.

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False
)

# Define a pose extraction function
def extract_landmarks(results):
    def lm_to_array(landmarks, n):
        if landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        else:
            return np.zeros((n, 3))

    pose = lm_to_array(results.pose_landmarks, 33)
    left_hand = lm_to_array(results.left_hand_landmarks, 21)
    right_hand = lm_to_array(results.right_hand_landmarks, 21)
    face = lm_to_array(results.face_landmarks, 468)

    return {
        "pose": pose,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "face": face
    }

# Process the video frame-by-frame
cap = cv2.VideoCapture(video_path)

pose_sequence = []

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic.process(rgb)

    landmarks = extract_landmarks(results)
    pose_sequence.append(landmarks)

cap.release()

print("Total frames processed:", len(pose_sequence))

# Inspect the output (sanity check)

sample = pose_sequence[0]

print("Pose shape:", sample["pose"].shape)          # (33, 3)
print("Left hand shape:", sample["left_hand"].shape) # (21, 3)
print("Right hand shape:", sample["right_hand"].shape)
print("Face shape:", sample["face"].shape)

# Minimal stick-figure visualization

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        frame,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        frame,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )

    cv2.imshow("Pose Check", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


