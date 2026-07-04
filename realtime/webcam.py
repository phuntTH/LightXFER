import os
import sys
import time
import threading
import cv2
import numpy as np
import torch
import mediapipe as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from multitask.multitask_model import MultiTaskFER
from robust_data.dataset import EMOTIONS, FEATURE_COLUMNS, get_eval_transform
from training.predict import load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOOTH_WINDOW = 5 
EMOTION_NAMES = [e.capitalize() for e in EMOTIONS]
FEATURE_NAMES = [f.replace("_", " ").title() for f in FEATURE_COLUMNS]
_eval_transform = get_eval_transform()

class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.stream.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()


def preprocess_face(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    transformed = _eval_transform(image=face_rgb)
    tensor = transformed["image"] 
    return tensor.unsqueeze(0)


def main():
    print(f"🚀 Đang khởi tạo mô hình trên thiết bị: {DEVICE.upper()}")
    model = load_model()

    if DEVICE == "cuda":
        model = model.half()
    
    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(min_detection_confidence=0.6, model_selection=0)

    vs = WebcamStream(src=0).start()
    time.sleep(1.0)
    
    emotion_buffer = []
    feature_buffer = []
    
    prev_time = 0
    fps = 0
    
    print("🎬 Hệ thống sẵn sàng với bộ lọc Temporal Smoothing! Nhấn 'q' để thoát.")

    while True:
        grabbed, frame = vs.read()
        if not grabbed or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bbox_data = detection.location_data.relative_bounding_box
                
                x = int(bbox_data.xmin * w)
                y = int(bbox_data.ymin * h)
                box_w = int(bbox_data.width * w)
                box_h = int(bbox_data.height * h)
                
                pad_w = int(box_w * 0.1)
                pad_h = int(box_h * 0.1)
                
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(w, x + box_w + pad_w)
                y2 = min(h, y + box_h + pad_h)

                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    continue

                face_crop = frame[y1:y2, x1:x2]
                
                input_tensor = preprocess_face(face_crop).to(DEVICE)
                if DEVICE == "cuda":
                    input_tensor = input_tensor.half()

                with torch.no_grad():
                    outputs = model(input_tensor)
                    
                    raw_emo_logits = outputs["emotion_logits"][0].float().cpu().numpy()
                    raw_feat_probs = torch.sigmoid(outputs["feature_logits"])[0].float().cpu().numpy()

                emotion_buffer.append(raw_emo_logits)
                feature_buffer.append(raw_feat_probs)

                if len(emotion_buffer) > SMOOTH_WINDOW:
                    emotion_buffer.pop(0)
                    feature_buffer.pop(0)

                avg_emo_logits = np.mean(emotion_buffer, axis=0)
                avg_feat_probs = np.mean(feature_buffer, axis=0)

                pred_emo_idx = np.argmax(avg_emo_logits)
                
                exp_logits = np.exp(avg_emo_logits - np.max(avg_emo_logits))
                emotion_prob = (exp_logits / np.sum(exp_logits))[pred_emo_idx] * 100
                emotion_text = f"{EMOTION_NAMES[pred_emo_idx]} ({emotion_prob:.1f}%)"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, emotion_text, (x1, max(15, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                offset_y = y1 + 20
                for idx, prob in enumerate(avg_feat_probs):
                    if prob > 0.5:
                        feat_text = f"- {FEATURE_NAMES[idx]}: {prob*100:.0f}%"
                        cv2.putText(frame, feat_text, (x2 + 10, offset_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                        offset_y += 18
                
                break
        else:
            emotion_buffer.clear()
            feature_buffer.clear()
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        cv2.putText(frame, f"System FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Multi-Task FER Live Demo (Smoothed)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\n🛑 Đang đóng hệ thống và giải phóng camera...")
    vs.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()