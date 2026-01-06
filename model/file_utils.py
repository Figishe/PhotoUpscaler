import os

def parse_image_file_paths(root):
    flie_paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith("._"):
                flie_paths.append(os.path.join(dirpath, f))
    return flie_paths