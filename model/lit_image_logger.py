import lightning as L
import torch
from torchvision.utils import make_grid
from model.image_utils import ycbcr_to_rgb_tensor

class ImageLoggerCallback(L.Callback):
    def __init__(self, val_samples, log_every_n_epochs=1):
        self.log_every_n_epochs = log_every_n_epochs
        self.fixed_batch = next(iter(val_samples))  # HR batch

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.log_every_n_epochs != 0:
            return

        pl_module.eval()

        with torch.no_grad():
            x, y = pl_module.batch_preprocess(self.fixed_batch)
            pred = pl_module(x)

        y_rgb = ycbcr_to_rgb_tensor(y.cpu())
        pred_rgb = ycbcr_to_rgb_tensor(pred.cpu())
        imgs = torch.cat([y_rgb, pred_rgb], dim=2)

        grid = make_grid(imgs, nrow=4)

        pl_module.logger.experiment.add_image(
            "GT_vs_Pred", grid, global_step=epoch
        )