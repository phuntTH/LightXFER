import os
import sys
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robust_data.dataset import create_dataloaders, EMOTIONS, FEATURE_COLUMNS
from multitask.multitask_model import MultiTaskFER

SPLITS_DIR = os.path.join(os.path.dirname(__file__), "data_preprocessing", "splits")
TRAIN_CSV = os.path.join(SPLITS_DIR, "train.csv")
VAL_CSV = os.path.join(SPLITS_DIR, "val.csv")
TEST_CSV = os.path.join(SPLITS_DIR, "test.csv")

BATCH_SIZE = 64
MODEL_PATH = os.path.join("checkpoints", "best_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DIAGNOSTICS_DIR = "diagnostics_report"
ERROR_IMAGES_DIR = os.path.join(DIAGNOSTICS_DIR, "misclassified_images")

EMOTION_NAMES = [e.capitalize() for e in EMOTIONS]
FEATURE_NAMES = [f.replace("_", " ").title() for f in FEATURE_COLUMNS]


# --- TÍCH HỢP CLASS MULTITASK GRAD-CAM ĐỂ GIẢI THÍCH MÔ HÌNH ---
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


# --- TÍCH HỢP HÀM TÍNH TOÁN ALIGNMENT SCORE TỰ ĐỘNG ---
def calculate_alignment_score(heatmap, true_features):
    """
    Tính toán xem năng lượng của Heatmap có phân bổ đúng vào vùng cơ mặt đích không.
    true_features: mảng nhị phân các thuộc tính cơ mặt của ảnh hiện tại.
    """
    h, w = heatmap.shape
    total_energy = np.sum(heatmap)
    if total_energy == 0:
        return 1.0

    # Khởi tạo các vùng mặt nạ thô hình học (Tỷ lệ chuẩn hóa trên khuôn mặt ảnh 224x224)
    brows_mask = np.zeros((h, w))
    brows_mask[int(h*0.15):int(h*0.45), int(w*0.15):int(w*0.85)] = 1  # Vùng mắt & lông mày
    
    mouth_mask = np.zeros((h, w))
    mouth_mask[int(h*0.6):int(h*0.9), int(w*0.2):int(w*0.8)] = 1     # Vùng miệng & cằm

    # Kiểm tra các nhóm thuộc tính tương ứng đang kích hoạt (true_features == 1)
    is_mouth_active = (true_features[0] == 1 or true_features[1] == 1 or true_features[7] == 1)
    is_brows_active = (true_features[2] == 1 or true_features[3] == 1 or true_features[4] == 1 or true_features[5] == 1 or true_features[6] == 1)

    if is_mouth_active and not is_brows_active:
        return np.sum(heatmap * mouth_mask) / total_energy
    elif is_brows_active and not is_mouth_active:
        return np.sum(heatmap * brows_mask) / total_energy
    elif is_mouth_active and is_brows_active:
        combined_mask = np.clip(mouth_mask + brows_mask, 0, 1)
        return np.sum(heatmap * combined_mask) / total_energy

    return 1.0


def save_confusion_matrix_plot(cm, target_names, output_path):
    """Vẽ và lưu ma trận nhầm lẫn thành file ảnh trực quan"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Emotion Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"💾 Đã lưu biểu đồ ma trận nhầm lẫn tại: {output_path}")

def denormalize_and_save(image_tensor, output_path):
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    plt.imsave(output_path, img)

def main():
    print(f"🚀 Đang khởi chạy hệ thống chẩn đoán trên thiết bị: {DEVICE.upper()}")
    
    if os.path.exists(DIAGNOSTICS_DIR):
        shutil.rmtree(DIAGNOSTICS_DIR)
    os.makedirs(ERROR_IMAGES_DIR, exist_ok=True)

    loaders = create_dataloaders(
        train_csv=TRAIN_CSV, val_csv=VAL_CSV, test_csv=TEST_CSV, batch_size=BATCH_SIZE
    )
    test_loader = loaders[2]
    
    model = MultiTaskFER()
    model = model.to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Lỗi: Không tìm thấy file {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    model.eval()

    # Khởi tạo bộ trích xuất Grad-CAM tự động đăng ký vào khối cuối backbone
    target_layer = model.backbone.backbone.blocks[-1]
    cam_extractor = MultiTaskGradCAM(model, target_layer)

    all_emo_preds, all_emo_labels = [], []
    all_feat_preds, all_feat_labels = [], []
    alignment_scores = []
    
    error_count = 0
    max_error_images = 100

    print("\n🔍 Đang thực hiện Inference và phân tích lỗi tự động...")
    # Loại bỏ block torch.no_grad() toàn cục bao quanh vòng lặp để Grad-CAM tính toán backward linh hoạt
    for batch in tqdm(test_loader, desc="Diagnostic Processing"):
        images = batch["image"].to(DEVICE, non_blocking=True)
        emotion_labels = batch["emotion"].to(DEVICE, non_blocking=True)
        feature_labels = batch["features"].to(DEVICE, non_blocking=True)

        # 1. Chạy Inference chính cực nhanh dưới cơ chế no_grad()
        with torch.no_grad():
            outputs = model(images)
            emotion_out = outputs["emotion_logits"]
            feature_out = outputs["feature_logits"]

            emo_preds = torch.argmax(emotion_out, dim=1)
            feat_preds = (torch.sigmoid(feature_out) > 0.5).int()

        for i in range(images.size(0)):
            true_emo = emotion_labels[i].item()
            pred_emo = emo_preds[i].item()
            
            if true_emo != pred_emo and error_count < max_error_images:
                folder_name = f"{EMOTION_NAMES[true_emo]}_but_predicted_{EMOTION_NAMES[pred_emo]}"
                specific_dir = os.path.join(ERROR_IMAGES_DIR, folder_name)
                os.makedirs(specific_dir, exist_ok=True)
                
                img_path = os.path.join(specific_dir, f"error_{error_count}.png")
                denormalize_and_save(images[i], img_path)
                error_count += 1

        all_emo_preds.extend(emo_preds.cpu().numpy())
        all_emo_labels.extend(emotion_labels.cpu().numpy())
        all_feat_preds.extend(feat_preds.cpu().numpy())
        all_feat_labels.extend(feature_labels.cpu().numpy())

        # 2. Tính toán Alignment Score cho ảnh đầu tiên của mỗi batch để theo dõi định lượng giải thích thị giác
        if images.size(0) > 0:
            single_img = images[0].unsqueeze(0)
            true_emo_idx = emotion_labels[0].item()
            true_feats = feature_labels[0].cpu().numpy()
            
            heatmap = cam_extractor.generate_heatmap(single_img, task_key="emotion_logits", class_idx=true_emo_idx)
            score = calculate_alignment_score(heatmap, true_feats)
            alignment_scores.append(score)

    # Giải phóng các hooks bảo vệ tài nguyên bộ nhớ VRAM
    cam_extractor.remove_hooks()

    all_emo_preds, all_emo_labels = np.array(all_emo_preds), np.array(all_emo_labels)
    all_feat_preds, all_feat_labels = np.array(all_feat_preds), np.array(all_feat_labels)

    report_path = os.path.join(DIAGNOSTICS_DIR, "detailed_metrics_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("      BÁO CÁO PHÂN TÍCH CHẨN ĐOÁN MÔ HÌNH TOÀN DIỆN\n")
        f.write("============================================================\n\n")
        
        # Ghi thêm kết quả định lượng XAI vào đầu báo cáo văn bản
        if len(alignment_scores) > 0:
            mean_alignment = np.mean(alignment_scores) * 100
            f.write("📊 CHỈ SỐ GIẢI THÍCH THỊ GIÁC (XAI METRICS):\n")
            f.write(f" -> XAI Alignment Score trung bình: {mean_alignment:.2f}%\n\n")
        
        f.write("1. ĐÁNH GIÁ CHI TIẾT NHÁNH EMOTION HEAD:\n")
        emo_report = classification_report(all_emo_labels, all_emo_preds, target_names=EMOTION_NAMES, digits=4)
        f.write(emo_report)
        
        f.write("\n2. ĐÁNH GIÁ CHI TIẾT NHÁNH MULTI-LABEL FEATURE HEAD:\n")
        feat_report = classification_report(all_feat_labels, all_feat_preds, target_names=FEATURE_NAMES, digits=4)
        f.write(feat_report)
        
    print(f"\n📝 Đã xuất file báo cáo văn bản chi tiết tại: {report_path}")

    cm = confusion_matrix(all_emo_labels, all_emo_preds)
    cm_img_path = os.path.join(DIAGNOSTICS_DIR, "emotion_confusion_matrix.png")
    save_confusion_matrix_plot(cm, EMOTION_NAMES, cm_img_path)
    print("\n=== KẾT QUẢ ĐÁNH GIÁ NHANH TRÊN TẬP TEST ===")
    print("Nhánh Emotion Report:")
    print(emo_report)
    print("--------------------------------------------------")
    if len(alignment_scores) > 0:
        print(f"📊 XAI Alignment Score trung bình: {np.mean(alignment_scores) * 100:.2f}%")
        print("--------------------------------------------------")
    print(f"💡 ĐÃ TRÍCH XUẤT XONG {error_count} ẢNH ĐOÁN SAI vào thư mục '{ERROR_IMAGES_DIR}'")
    print("Hãy mở thư mục này để phân tích trực quan nguyên nhân lỗi của mô hình!")

if __name__ == "__main__":
    main()