# process_dev.py
import gzip
import pickle
import os
from tqdm import tqdm

from pose_extractor import PoseExtractor
from pose_normalization import normalize_pose_sequence
from gloss_alignment import uniform_align
from storage import init_gloss_dict, save_gloss_pose_dict

ANNOT_PATH = "phoenix14t.pami0.dev.annotations_only.gzip"
VIDEO_ROOT = "videos"
OUT_PATH = "output/gloss_pose_dev.pkl"

# Load annotations
with gzip.open(ANNOT_PATH, "rb") as f:
    annotations = pickle.load(f)

extractor = PoseExtractor()
gloss_pose_dict = init_gloss_dict()

processed = 0
skipped = 0

for ann in tqdm(annotations):
    video_path = os.path.join(VIDEO_ROOT, f"{ann['name']}.mp4")

    if not os.path.exists(video_path):
        skipped += 1
        continue

    pose_seq = extractor.extract_video(video_path)
    if pose_seq is None or len(pose_seq) < 10:
        skipped += 1
        continue

    pose_seq = normalize_pose_sequence(pose_seq)
    segments = uniform_align(pose_seq, ann["gloss"])

    for gloss, segment in segments:
        if len(segment) > 0:
            gloss_pose_dict[gloss].append(segment)

    processed += 1

print(f"Processed: {processed}")
print(f"Skipped: {skipped}")
print(f"Unique glosses: {len(gloss_pose_dict)}")

save_gloss_pose_dict(gloss_pose_dict, OUT_PATH)
print(f"Saved → {OUT_PATH}")
