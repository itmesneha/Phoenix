# pose_normalization.py
import numpy as np

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

def normalize_pose_sequence(pose_seq):
    """
    pose_seq: (T, 75, 3)
    """
    normalized = []

    for frame in pose_seq:
        body = frame[:33]
        hands = frame[33:]

        root = (body[LEFT_HIP] + body[RIGHT_HIP]) / 2
        body -= root
        hands -= root

        scale = np.linalg.norm(
            body[LEFT_SHOULDER] - body[RIGHT_SHOULDER]
        ) + 1e-6

        body /= scale
        hands /= scale

        normalized.append(np.concatenate([body, hands], axis=0))

    return np.stack(normalized)
