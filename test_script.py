import pickle
import gzip
import os
import cv2
import mediapipe as mp
import numpy as np


with gzip.open('phoenix14t.pami0.dev.annotations_only.gzip', 'rb') as f:
    annotations = pickle.load(f)

print('type: ', type(annotations))
annotation = annotations[0]
print(type(annotation))
print(annotation.keys())
print('name: ', annotation['name'])
print('gloss: ', annotation['gloss'])

video_path = f"videos/{annotation['name']}.mp4"
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

# till here we have:

# ✅ pose_sequence: frame-by-frame motion

# ✅ sample['gloss']: gloss string sequence

# ❌ Missing: which frames correspond to which gloss

# alignment strategy - Uniform temporal segmentation

gloss_tokens = annotation['gloss'].split()
print(gloss_tokens)

# Split pose sequence uniformly
T = len(pose_sequence)
G = len(gloss_tokens)

frames_per_gloss = T // G

# Build gloss → pose segments
gloss_pose_pairs = []

for i, gloss in enumerate(gloss_tokens):
    start = i * frames_per_gloss
    end = (i + 1) * frames_per_gloss if i < G - 1 else T

    segment = pose_sequence[start:end]

    gloss_pose_pairs.append({
        "gloss": gloss,
        "pose": segment
    })

# Convert pose segment → tensor

def stack_pose(segment):
    pose = np.stack([f["pose"] for f in segment])           # (Tg, 33, 3)
    lh = np.stack([f["left_hand"] for f in segment])        # (Tg, 21, 3)
    rh = np.stack([f["right_hand"] for f in segment])       # (Tg, 21, 3)

    return np.concatenate([pose, lh, rh], axis=1)           # (Tg, 75, 3)

for item in gloss_pose_pairs:
    item["pose_tensor"] = stack_pose(item["pose"])

# save one sample

np.save(
    "sample_gloss_pose.npy",
    {
        "gloss": gloss_pose_pairs[0]["gloss"],
        "pose": gloss_pose_pairs[0]["pose_tensor"]
    }
)

# till here we have:
# PHOENIX video
#    ↓
# MediaPipe pose
#    ↓
# Frame-wise motion
#    ↓
# Gloss-aligned pose segments   

# normalization function
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

def normalize_pose(pose_tensor):
    """
    pose_tensor: (T, 75, 3)
    returns: normalized pose tensor (T, 75, 3)
    """
    normalized = []

    for frame in pose_tensor:
        body = frame[:33]
        hands = frame[33:]  # hands already stacked

        # Root (mid-hip)
        root = (body[LEFT_HIP] + body[RIGHT_HIP]) / 2

        # Center body + hands
        body = body - root
        hands = hands - root

        # Scale using shoulder width
        shoulder_dist = np.linalg.norm(
            body[LEFT_SHOULDER] - body[RIGHT_SHOULDER]
        ) + 1e-6

        body /= shoulder_dist
        hands /= shoulder_dist

        normalized.append(np.concatenate([body, hands], axis=0))

    return np.stack(normalized)


item = gloss_pose_pairs[0]
normalized_pose = normalize_pose(item["pose_tensor"])

print("Normalized shape:", normalized_pose.shape)

# sanity check 
import matplotlib.pyplot as plt

frame = normalized_pose[0]

x = frame[:, 0]
y = -frame[:, 1]  # invert Y for visualization

plt.figure(figsize=(4,6))
plt.scatter(x[:33], y[:33])   # body
plt.scatter(x[33:], y[33:], s=5)  # hands
plt.title("Normalized Pose")
plt.axis("equal")
plt.show()

