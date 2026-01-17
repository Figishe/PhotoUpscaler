from numpy import size
import torch
import torchvision.io as tvio
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image

from model.dataset import SuperResDataset

from model.image_utils import rgb_to_ycbcr_tensor, ycbcr_tensor_to_pil

class Inference():
    
    def calc_blend_mask(self, size: int) -> torch.Tensor:
        patch_blend_mask = torch.zeros(
            (3, size, size),
            device=self.device
        )
        blend_margin = self.OVERLAP * self.UPSCALE_FACTOR
        for i in range(size):
            for j in range(size):
                wx = 1.0
                wy = 1.0
                if i < blend_margin:
                    wx = i / blend_margin
                elif i >= size - blend_margin:
                    wx = (size - i - 1) / blend_margin
                if j < blend_margin:
                    wy = j / blend_margin
                elif j >= size - blend_margin:
                    wy = (size - j - 1) / blend_margin
                w = 0.1 + wx * wy
                patch_blend_mask[:, i, j] = w
        
        return patch_blend_mask


    def __init__(self, model: torch.nn.Module, batch_size: int=32) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.batch_size = batch_size

        self.PATCH_SIZE = 64
        self.OVERLAP = self.PATCH_SIZE // 8
        self.STRIDE = self.PATCH_SIZE - self.OVERLAP
        self.UPSCALE_FACTOR = 2 # TODO: take from model
        self.NUM_CHANNELS = 3
        self.PATCH_SIZE_UPSCALED = self.PATCH_SIZE * self.UPSCALE_FACTOR

        self.patch_blend_mask = self.calc_blend_mask(self.PATCH_SIZE_UPSCALED)


    def cut_patches(self, img: torch.Tensor, size: int, stride: int) -> tuple[torch.Tensor, int, int]:
        # (B, C, H, W)
        DIM_C = len(img.shape) - 3
        DIM_H = len(img.shape) - 2
        DIM_W = len(img.shape) - 1

        patches = img.unfold(DIM_H, size, stride)\
                     .unfold(DIM_W, size, stride)
        # patches: (B, C, nH, nW, size, size)

        patches = patches.permute(0, 2, 3, 1, 4, 5)  # (B, nH, nW, C, size, size)
        B, nH, nW, C, H_patch, W_patch = patches.shape
        patches = patches.contiguous().view(B * nH * nW, C, H_patch, W_patch)

        return patches, nH, nW


    def predict_patches(self, patches: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        preds = []
        
        with torch.no_grad():
            for i in range(0, len(patches), self.batch_size):
                batch = patches[i:i + self.batch_size].to(self.device)
                out = self.model(batch).clone()
                preds.append(out)

        patches_pred = torch.cat(preds, dim=0)
        return patches_pred


    def assemble_patches(self, size: int, stride: int, n_patches_h: int, n_patches_w: int, patches_pred: torch.Tensor) -> torch.Tensor:
        H_out_up = ((n_patches_h - 1) * stride + size) * self.UPSCALE_FACTOR
        W_out_up = ((n_patches_w - 1) * stride + size) * self.UPSCALE_FACTOR
        
        out_img = torch.zeros((3, H_out_up, W_out_up), device=self.device)
        out_weight = torch.zeros_like(out_img)

        overlap_up = self.OVERLAP * self.UPSCALE_FACTOR
        stride_up = size - overlap_up

        for iy in range(n_patches_h):
            for ix in range(n_patches_w):
                idx = iy * n_patches_w + ix  # same as after permute+view in cut_patches
                x = ix * stride_up
                y = iy * stride_up
                patch_h, patch_w = patches_pred[idx].shape[-2:]
                mask_patch = self.patch_blend_mask[:, :patch_h, :patch_w]
                out_img[:, y:y+patch_h, x:x+patch_w] += patches_pred[idx] * mask_patch
                out_weight[:, y:y+patch_h, x:x+patch_w] += mask_patch

        out_img = out_img / out_weight.clamp(min=1e-6)
        return out_img


    def upscale(self, img_pil: Image.Image) -> Image.Image:
        img = TF.pil_to_tensor(img_pil)
        img = img[:self.NUM_CHANNELS, :, :]  # drop alpha if presented
        img = img.float() / 255.0
        img = rgb_to_ycbcr_tensor(img)
        
        h, w = img.shape[-2:]
        img = img.unsqueeze(0)  # add batch dim

        stride = self.STRIDE

        tail_h = (h - self.PATCH_SIZE) % stride
        pad_h = 0 if tail_h == 0 else stride - tail_h

        tail_w = (w - self.PATCH_SIZE) % stride
        pad_w = 0 if tail_w == 0 else stride - tail_w

        img = F.pad(
            img,
            (0, pad_w, 0, pad_h), 
            mode="reflect"
        )

        patches, n_patches_h, n_patches_w = self.cut_patches(img, self.PATCH_SIZE, self.STRIDE)
        patches_pred = self.predict_patches(patches)
        out_img = self.assemble_patches(self.PATCH_SIZE_UPSCALED, self.STRIDE, n_patches_h, n_patches_w, patches_pred)
        
        # remove padding
        out_img = out_img[:, : (h * self.UPSCALE_FACTOR), : (w * self.UPSCALE_FACTOR)]

        return ycbcr_tensor_to_pil(out_img.cpu())



