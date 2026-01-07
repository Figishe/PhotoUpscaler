import torch
import torchvision.io as tvio
import torchvision.transforms.functional as TF

from model.dataset import SuperResDataset

from model.image_utils import ycbcr_tensor_to_pil

class Inference():
    
    def __init__(self, model, batch_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.batch_size = batch_size

    def upscale(self, img_pil):
        img = TF.pil_to_tensor(img_pil)
        img = img[:3, :, :]  # drop alpha if presented
        img = img.float() / 255.0
        img = img * 2 - 1
        
        h, w = img.shape[1], img.shape[2]
        # split into patches of size 64x64
        img = img.unsqueeze(0)  # add batch dim
        patches = []
        PATCH_SIZE = 64

        ny = h // PATCH_SIZE
        nx = w // PATCH_SIZE

        patches = []
        for iy in range(ny):
            for ix in range(nx):
                y = iy * PATCH_SIZE
                x = ix * PATCH_SIZE
                patch = img[:, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patches.append(patch)

        patches = torch.cat(patches, dim=0)

        self.model.eval()
        preds = []
        
        with torch.no_grad():
            for i in range(0, len(patches), self.batch_size):
                batch = patches[i:i + self.batch_size].to(self.device)
                out = self.model(batch)
                preds.append(out.cpu())

        patches_pred = torch.cat(preds, dim=0)

        
        # reconstruct image from patches
        UPSCALE_FACTOR = 2 # TODO: take from model
        PATCH_SIZE_UPSCALED = PATCH_SIZE * 2
        
        out_img = torch.zeros(
            (3, ny * PATCH_SIZE_UPSCALED, nx * PATCH_SIZE_UPSCALED),
            device=self.device
        )

        idx = 0
        for iy in range(ny):
            for ix in range(nx):
                y = iy * PATCH_SIZE_UPSCALED
                x = ix * PATCH_SIZE_UPSCALED
                out_img[:, y:y+PATCH_SIZE_UPSCALED, x:x+PATCH_SIZE_UPSCALED] = patches_pred[idx]
                idx += 1
        
        return ycbcr_tensor_to_pil(out_img.cpu())