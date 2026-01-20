# gloss_alignment.py
def uniform_align(pose_seq, gloss_str):
    """
    pose_seq: (T,75,3)
    gloss_str: "WETTER MORGEN REGNERISCH"
    """
    tokens = gloss_str.split()
    T = len(pose_seq)
    G = len(tokens)

    frames_per_gloss = T // G
    segments = []

    for i, gloss in enumerate(tokens):
        start = i * frames_per_gloss
        end = T if i == G - 1 else (i + 1) * frames_per_gloss

        segments.append((gloss, pose_seq[start:end]))

    return segments
