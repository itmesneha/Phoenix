# pose_extractor.py
import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic

class PoseExtractor:
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False
        )

    def _lm_to_array(self, landmarks, n):
        if landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        return np.zeros((n, 3))

    def extract_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        pose = self._lm_to_array(results.pose_landmarks, 33)
        lh   = self._lm_to_array(results.left_hand_landmarks, 21)
        rh   = self._lm_to_array(results.right_hand_landmarks, 21)

        return np.concatenate([pose, lh, rh], axis=0)  # (75,3)

    def extract_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            sequence.append(self.extract_frame(frame))

        cap.release()
        return np.stack(sequence) if len(sequence) > 0 else None
