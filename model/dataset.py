import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class SuperResDataset(Dataset):

    def __init__(self, root, crop_size=192):
        self.crop_size = crop_size

        self.paths = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith((".jpg", ".jpeg")) and not f.startswith("._"):
                    self.paths.append(os.path.join(dirpath, f))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        w, h = img.size

        left = random.randint(0, w - self.crop_size)
        top  = random.randint(0, h - self.crop_size)

        Y = img.crop((left, top,
                      left + self.crop_size,
                      top + self.crop_size))

        return self._pil_to_tensor(Y)


    def _pil_to_tensor(self, img):
        t = torch.from_numpy(
            np.array(img, dtype=np.uint8)
        ).permute(2, 0, 1).float() / 255.0

        t = t * 2 - 1   # [-1, 1]
        return t
    

    def tensor_to_pil(self, tensor):
        t = tensor.clone()
        t = torch.clamp(t, -1.0, 1.0)

        arr = t.permute(1, 2, 0) # CxHxW -> HxWxC
        arr = (arr + 1.0) / 2.0 # [0; 1]
        arr = arr * 255 # [0; 255]
        arr = arr.cpu().numpy()
        arr = arr.astype(np.uint8)

        img_rgb = Image.fromarray(arr, mode="RGB")

        return img_rgb

