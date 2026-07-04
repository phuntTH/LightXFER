import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from multitask.multitask_model import MultiTaskFER

# Configuration setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/best_model.pth" 
IMAGE_PATH = "D:/LightXFER/dataset_final/test/angry/affectnet_0096114.jpg"              
OUTPUT_DIR = "diagnostics_report/gradcam_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class names lists
EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
FEATURE_NAMES = ["Smile", "Mouth Open", "Eyes Wide Open", "Eyes Narrowed", 
                 "Brows Together", "Lowered Eyebrows", "Raised Eyebrows", "Compressed Lips"]

# --- Vectorized MultiTaskGradCAM Class ---
class MultiTaskGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self.hooks.append(self.target_layer.register_forward_hook(self._save_activation))
        # Use register_full_backward_hook to avoid deprecation warnings and ensure stability
        self.hooks.append(self.target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output.detach().clone()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach().clone()

    def generate_heatmap(self, input_tensor, task_key="emotion_logits", class_idx=0):
        with torch.enable_grad():
            self.model.zero_grad()
            outputs = self.model(input_tensor)
            logits = outputs[task_key]
            target_score = logits[0, class_idx]
            target_score.backward(retain_graph=False)
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze(0)
        heatmap = torch.nn.functional.relu(heatmap)
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


# --- MAIN PIPELINE ---
def main():
    # 1. Initialize and load model checkpoints
    print(f"[*] Loading model onto {DEVICE}...")
    model = MultiTaskFER().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"[+] Loaded checkpoint successfully: {MODEL_PATH}")
    else:
        print(f"[!] Checkpoint not found at {MODEL_PATH}. Running with random weights.")
        return
    model.eval()

    # Target Layer identification
    target_layer = model.backbone.backbone.blocks[-1]
    cam_extractor = MultiTaskGradCAM(model, target_layer)

    # 2. Image loading and preprocessing
    if not os.path.exists(IMAGE_PATH):
        print(f"[!] Image not found at {IMAGE_PATH}.")
        return
        
    orig_img = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

    # 3. DYNAMIC PREDICTION (Forward Pass without gradients to decide targets)
    print("[*] Running image inference for dynamic target selection...")
    with torch.no_grad():
        outputs = model(input_tensor)
        emotion_logits = outputs["emotion_logits"]
        feature_logits = outputs["feature_logits"]

        # Dynamic selection for Emotion (Argmax)
        pred_emotion_idx = torch.argmax(emotion_logits, dim=1).item()
        
        # Dynamic selection for Features (Sigmoid threshold > 0.5)
        feature_probs = torch.sigmoid(feature_logits[0])
        pred_feature_indices = torch.where(feature_probs > 0.5)[0].tolist()
        
        # Fallback: If no feature exceeds 50%, select the highest one to guarantee a visualization
        if not pred_feature_indices:
            pred_feature_indices = [torch.argmax(feature_logits, dim=1).item()]

    print("-" * 50)
    print(f"[➔] Model Predicted Emotion: '{EMOTION_NAMES[pred_emotion_idx].upper()}'")
    print(f"[➔] Model Detected Features: {[FEATURE_NAMES[idx] for idx in pred_feature_indices]}")
    print("-" * 50)

    # 4. Generate Heatmap for the Predicted Emotion
    print(f"[*] Generating Grad-CAM for Predicted Emotion: '{EMOTION_NAMES[pred_emotion_idx].upper()}'...")
    heatmap_emo = cam_extractor.generate_heatmap(input_tensor, task_key="emotion_logits", class_idx=pred_emotion_idx)
    result_emo = overlay_heatmap(heatmap_emo, orig_img, alpha=0.4)
    
    emo_out_path = os.path.join(OUTPUT_DIR, f"predicted_emotion_{EMOTION_NAMES[pred_emotion_idx]}.png")
    cv2.imwrite(emo_out_path, result_emo)
    print(f"[+] Saved Emotion Heatmap at: {emo_out_path}\n")

    # 5. Generate Heatmaps for all triggered Features
    for feat_idx in pred_feature_indices:
        prob = feature_probs[feat_idx].item()
        feat_name = FEATURE_NAMES[feat_idx]
        print(f"[*] Generating Grad-CAM for Active Feature: '{feat_name.upper()}' (Confidence: {prob*100:.2f}%)...")
        
        heatmap_feat = cam_extractor.generate_heatmap(input_tensor, task_key="feature_logits", class_idx=feat_idx)
        result_feat = overlay_heatmap(heatmap_feat, orig_img, alpha=0.4)
        
        file_safe_name = feat_name.lower().replace(" ", "_")
        feat_out_path = os.path.join(OUTPUT_DIR, f"predicted_feature_{file_safe_name}.png")
        cv2.imwrite(feat_out_path, result_feat)
        print(f"[+] Saved Feature Heatmap at: {feat_out_path}")

    # Cleanup memory hooks
    cam_extractor.remove_hooks()
    print("\n[+] Dynamic execution finished successfully.")

if __name__ == "__main__":
    main()