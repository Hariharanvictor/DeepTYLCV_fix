import torch
import torch.nn.functional as F
import lightning as pl
from .architecture import DeepTYLCV_CONV
from .loss import FocalLoss
from .metrics import calculate_metrics
import numpy as np

    
class DeepTYLCV_Module(pl.LightningModule):
    def __init__(self, model_config, trainer_config, loss_config):
        super(DeepTYLCV_Module, self).__init__()
        self.model = DeepTYLCV_CONV(**vars(model_config))
        self.loss_fn = FocalLoss(alpha=torch.tensor(loss_config.alpha), gamma=loss_config.gamma)
        
        self.trainer_config = trainer_config
    
    def forward(self, X, masks_X,X_conv):
        return self.model(X, masks_X, X_conv)
    
    def training_step(self, batch, batch_idx):
        X,X_conv, masks_X, y = batch
        y_logits = self.forward(X, masks_X,X_conv)
        y_hat = F.softmax(y_logits, dim=-1)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def on_validation_epoch_start(self):
        self.all_y = []
        self.all_y_hat = []
        self.all_y_probs = []
    
    def validation_step(self, batch, batch_idx):
        X,X_conv, masks_X, y = batch
        y_logits = self.forward(X, masks_X,X_conv)
        y_probs = F.softmax(y_logits, dim=-1)
        
        # Get the probabilities of the predicted class, if class 0, reverse the probability by choosing the second prob only
        y_probs_reversed = y_probs[:, 1]
        
        # Store for calculating metric later
        self.all_y.append(y)
        self.all_y_probs.append(y_probs_reversed)
        
    def on_validation_epoch_end(self):
        all_y = torch.cat(self.all_y).detach().cpu().numpy()
        all_y_probs = torch.cat(self.all_y_probs).detach().cpu().numpy()
        
        best_bacc = -100
        best_metrics = {}
        metrics = calculate_metrics(all_y_probs, all_y, 0.5)
        if metrics['bacc'] > best_bacc:
            metrics['threshold'] = 0.5
            best_bacc = metrics['bacc']
            best_metrics = metrics
        self.log_dict(best_metrics)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.trainer_config.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        return [optimizer], [scheduler] 