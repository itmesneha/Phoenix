import pickle
import gzip
import os


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


