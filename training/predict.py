import os
import sys
import cv2
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from multitask.multitask_model import MultiTaskFER
from robust_data.dataset import EMOTIONS, FEATURE_COLUMNS, get_eval_transform

EMOTION_NAMES = [e.capitalize() for e in EMOTIONS]
FEATURE_NAMES = [f.replace("_", " ").title() for f in FEATURE_COLUMNS]
MODEL_PATH = os.path.join("checkpoints", "best_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    print(f"📦 Đang khởi tạo mô hình trên thiết bị: {DEVICE.upper()}...")
    model = MultiTaskFER() 
    model = model.to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy file trọng số tối ưu tại: {MODEL_PATH}")
        
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    model.eval()    
    return model

def preprocess_image(image_path):
    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        raise ValueError(f"❌ Không thể đọc dữ liệu ảnh tại: {image_path}. Vui lòng kiểm tra lại đường dẫn!")
    
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    transform = get_eval_transform()
    transformed = transform(image=img_rgb)
    
    input_tensor = transformed["image"].unsqueeze(0).to(DEVICE)
    return img_cv2, input_tensor

def draw_results(image, emotion_label, confidence, active_features):
    output_img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(output_img, f"Emotion: {emotion_label} ({confidence:.1f}%)", 
                (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.putText(output_img, "Features:", (10, 70), font, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    y_offset = 100
    
    if len(active_features) == 0:
        cv2.putText(output_img, "- None", (10, y_offset), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        for feat in active_features:
            cv2.putText(output_img, f"- {feat}", (10, y_offset), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 30
            
    return output_img

def main(image_path):
    model = load_model()
    
    print(f"🔍 Đang tiến hành phân tích nhận diện: {image_path}")
    original_img, input_tensor = preprocess_image(image_path)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        emotion_out = outputs["emotion_logits"]
        feature_out = outputs["feature_logits"]
        emotion_probs = torch.softmax(emotion_out, dim=1)
        emotion_idx = torch.argmax(emotion_probs, dim=1).item()
        confidence = emotion_probs[0][emotion_idx].item() * 100
        predicted_emotion = EMOTION_NAMES[emotion_idx]
        
        feature_probs = torch.sigmoid(feature_out)[0]
        active_features = [FEATURE_NAMES[i] for i, prob in enumerate(feature_probs) if prob > 0.3]
    print("\n" + "="*45)
    print("🎯 KẾT QUẢ INFERENCE THÀNH CÔNG:")
    print(f" 🔹 Emotion chính : {predicted_emotion} ({confidence:.2f}%)")
    print(f" 🔹 Đặc trưng nền (Threshold > 0.5): {', '.join(active_features) if active_features else 'None'}")
    print(" 🔹 [Debug] Xác suất từng đặc trưng:")
    for i, prob in enumerate(feature_probs):
        print(f"     - {FEATURE_NAMES[i]}: {prob.item():.4f}")
    print("="*45 + "\n")

    result_img = draw_results(original_img, predicted_emotion, confidence, active_features)
    
    h, w = result_img.shape[:2]
    if h > 800 or w > 800:
        scale = 800 / max(h, w)
        result_img = cv2.resize(result_img, (int(w * scale), int(h * scale)))
        
    cv2.imshow("Multi-Task FER - Single Image Inference", result_img)
    print("📸 Cửa sổ ảnh đã mở. Nhấn bất kỳ phím nào trên bàn phím để kết thúc hàm.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    TEST_IMAGE = "D:/LightXFER/data/affectnet/test/Anger/image0000060.jpg" 
    
    if os.path.exists(TEST_IMAGE):
        main(TEST_IMAGE)
    else:
        print(f"⚠️ Cảnh báo: Vui lòng chuẩn bị một bức ảnh thực tế và gán lại biến TEST_IMAGE = '{TEST_IMAGE}' để chạy thử nghiệm.")