import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.io as tvio
from model.file_utils import parse_image_file_paths

class SuperResDataset(Dataset):

    current_image = None
    current_image_id = -1

    def __init__(self, 
                 root, 
                 patch_size=128, 
                 downscale=(2808, 1872), # half-res of Canon 5dII frame
                 seed=None,
                ):
        
        self.rnd = random.Random(seed)

        self.patch_size = patch_size

        self.flie_paths = parse_image_file_paths(root)

        image_w, image_h = downscale
        patches_per_image_w = image_w // patch_size
        patches_per_image_h = image_h // patch_size
        self.magrin_w = image_w - (patch_size * patches_per_image_w)
        self.magrin_h = image_h - (patch_size * patches_per_image_h)

        self.image_w = image_w
        self.image_h = image_h

        self.patches_per_image_w = patches_per_image_w
        self.patches_per_image_h = patches_per_image_h
        self.patches_per_image = patches_per_image_w * patches_per_image_h


    def __len__(self):
        return len(self.flie_paths) * self.patches_per_image


    def __getitem__(self, idx):
        image_id = idx // self.patches_per_image
        patch_id = idx % self.patches_per_image

        if image_id != self.current_image_id:
            # update cache
            img = tvio.read_image(self.flie_paths[image_id])
            img = img.unsqueeze(0)  # add batch dim (compatible with interpolate)
            img = torch.nn.functional.interpolate(img, size=(self.image_h, self.image_w), mode="bicubic", align_corners=False)
            img = img.squeeze(0)
            img = img.float() / 255.0
            img = img * 2 - 1
            self.current_image_id = image_id
            self.current_image = img

            # Randomize patch loading (because native shuffle=True will not work with the image cache)
            self.patch_indices = [(i,j) for i in range(self.patches_per_image_h) for j in range(self.patches_per_image_w)]
            self.rnd.shuffle(self.patch_indices)
        else:
            # continue taking patches from the cached image
            img = self.current_image
        
        grid_h, grid_w = self.patch_indices[patch_id]
        
        top  = grid_h * self.patch_size + self.rnd.randint(0, self.magrin_h)
        left = grid_w * self.patch_size + self.rnd.randint(0, self.magrin_w)

        patch = img[:, top:top+self.patch_size, left:left+self.patch_size]

        return patch
    

    @staticmethod
    def tensor_to_pil(tensor):
        t = tensor.clone()
        t = torch.clamp(t, -1.0, 1.0)

        arr = t.permute(1, 2, 0) # CxHxW -> HxWxC
        arr = (arr + 1.0) / 2.0 # [0; 1]
        arr = arr * 255 # [0; 255]
        arr = arr.cpu().numpy()
        arr = arr.astype(np.uint8)

        img_rgb = Image.fromarray(arr, mode="RGB")

        return img_rgb

