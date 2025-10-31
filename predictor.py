from src.architecture import CONTRA_IL6_CONV
import torch
from torch.nn.functional import softmax
import os
import numpy as np
from torch.nn import functional as F
class DeepTYLCV_Predictor:
    def __init__(self, model_config, ckpt_dir, nfold, device):
        self.models = [CONTRA_IL6_CONV(**model_config) for _ in range(nfold)]
        self.device = device
        
        ckpt_paths = []
        for file in sorted(os.listdir(ckpt_dir)):
            if file.endswith('.ckpt'):
                ckpt_paths.append(os.path.join(ckpt_dir, file))

        # for ckpt_file, model in zip(ckpt_paths, self.models):
        #     if not ckpt_file.endswith('.ckpt'):
        #         continue

            
            
        #     pt_file_path = os.path.join(ckpt_dir, ckpt_file)
        #     self._load_model_from_checkpoint(pt_file_path, model)

        for i in range(nfold):
            ckpt_file = ckpt_paths[i]
            pt_file_path = os.path.join(ckpt_dir, ckpt_file)
            # print(ckpt_file)
            self.models[i] = self._load_model_from_checkpoint(pt_file_path, self.models[i])
    
    def _load_model_from_checkpoint(self, checkpoint_path, model):
        state_dict = torch.load(checkpoint_path, map_location=self.device)['state_dict']
        # replace 'model.' prefix if exists
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        # remove 'loss_fn.' key if exists
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('loss_fn.')}
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(self.device)
        return model
    # import torch

    def predict_one(self, f1, f2, f3, fccd, threshold=0.5):
        # Convert numpy → torch if needed
        if isinstance(f1, np.ndarray): f1 = torch.from_numpy(f1)
        if isinstance(f2, np.ndarray): f2 = torch.from_numpy(f2)
        if isinstance(f3, np.ndarray): f3 = torch.from_numpy(f3)
        if isinstance(fccd, np.ndarray): fccd = torch.from_numpy(fccd)

        # Move to device
        f1 = f1.to(self.device)
        f2 = f2.to(self.device)
        f3 = f3.to(self.device)
        fccd = fccd.to(self.device)
        f1= F.pad(f1, (0, 0, 0, 363 - f1.size(0)), value=0)
        f2= F.pad(f2, (0, 0, 0, 363 - f2.size(0)), value=0)
        f3= F.pad(f3, (0, 0, 0, 363 - f3.size(0)), value=0)
        # print model devices
        # for model in self.models:
        #     print(next(model.parameters()).device)

        all_prob = []
        for model in self.models:
            with torch.no_grad():
                logit = model.forward([f1.unsqueeze(0), f2.unsqueeze(0), f3.unsqueeze(0)], None, fccd.unsqueeze(0))
                prob = torch.nn.functional.softmax(logit, dim=-1)[:, 1]
                all_prob.append(prob)

        all_prob = torch.stack(all_prob).squeeze(1)
        print(all_prob)
        is_severe = "Severe" if all_prob.mean() >= threshold else "Mild"

        return is_severe, all_prob.mean()

    

    def __call__(self, f1s, f2s, f3s, fccds, threshold=0.5):
        severity_predictions = []
        probs = []
        
        for f1, f2, f3, fccd in zip(f1s, f2s, f3s, fccds):
            is_severe, prob = self.predict_one(f1, f2, f3, fccd, threshold)
            severity_predictions.append(is_severe)
            probs.append(prob)
            
        return severity_predictions, probs

    