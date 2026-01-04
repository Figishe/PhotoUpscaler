import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class SuperResDataset(Dataset):

    def __init__(self, root, crop_size=192, downscale_denoise=4, downscale=2):
        self.crop_size = crop_size
        self.downscale = downscale
        self.downscale_denoise = downscale_denoise

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
        w //= self.downscale_denoise
        h //= self.downscale_denoise

        img_denoised = img.resize((w, h), Image.LANCZOS)

        left = random.randint(0, w - self.crop_size)
        top = random.randint(0, h - self.crop_size)
        Y = img_denoised.crop((left, top,
                      left + self.crop_size,
                      top + self.crop_size))

        X = Y.resize(
            (self.crop_size // self.downscale,
             self.crop_size // self.downscale),
            Image.BICUBIC
        )

        Y_tensor = self._pil_to_tensor(Y)
        X_tensor = self._pil_to_tensor(X)

        return X_tensor, Y_tensor

    def _pil_to_tensor(self, img):
        arr = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        arr = arr.view(img.size[1], img.size[0], 3)  # HxWxC
        arr = arr.permute(2, 0, 1).float() / 255.0  # CxHxW
        return arr
    

    def tensor_to_pil(self, tensor):
        t = tensor.clone()
        t = torch.clamp(t, 0.0, 1.0)

        # CxHxW -> HxWxC
        arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        img_rgb = Image.fromarray(arr, mode="sRGB")

        return img_rgb

