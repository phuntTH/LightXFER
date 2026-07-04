import cv2
import numpy as np
import torch
import torch.nn.functional as F

class MultiTaskGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []

        self.hooks.append(self.target_layer.register_forward_hook(self._save_activation))
        self.hooks.append(self.target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, task_key="emotion_logits", class_idx=0):
        with torch.enable_grad():
            self.model.zero_grad()
            
            outputs = self.model(input_tensor)
            logits = outputs[task_key]
            
            target_score = logits[0, class_idx]
            
            target_score.backward(retain_graph=False)
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze(0) 
        
        heatmap = F.relu(heatmap)
        
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val
        return heatmap.cpu().numpy()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def overlay_heatmap(heatmap, original_img, alpha=0.4):

    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    return cv2.addWeighted(original_img, 1.0 - alpha, heatmap_colored, alpha, 0)