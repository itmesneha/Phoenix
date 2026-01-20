# storage.py
import pickle
import os
from collections import defaultdict

def save_gloss_pose_dict(gloss_pose_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(gloss_pose_dict, f)

def load_gloss_pose_dict(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def init_gloss_dict():
    return defaultdict(list)
