import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robust_data.dataset import create_dataloaders
from robust_data.loss import MultiLabelFocalLoss
from multitask.multitask_model import MultiTaskFER
from tqdm import tqdm

from utils import EarlyStopping, print_epoch_result, save_checkpoint, load_checkpoint
from validate import validate

# =====================================
# CONFIGURATIONS
# =====================================
SPLITS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data_preprocessing", "splits"
)
TRAIN_CSV = os.path.join(SPLITS_DIR, "train.csv")
VAL_CSV = os.path.join(SPLITS_DIR, "val.csv")
TEST_CSV = os.path.join(SPLITS_DIR, "test.csv")

BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 3e-5
SAVE_DIR = "checkpoints"


def main():
    os.*makedirs(SAVE_DIR, exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        
    print(f"Khoi chay huan luyen tren thiet bi: {DEVICE.upper()}")

    # =====================================
    # DATA PIPELINE (PHASE 2 INTEGRATION)
    # =====================================
    (
        train_loader,
        val_loader,
        test_loader,
        _,
        _,
        _,
        class_weights_tensor,
    ) = create_dataloaders(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        batch_size=BATCH_SIZE,
        num_workers=2,
    )

    # =====================================
    # MODEL INITIALIZATION
    # =====================================
    model = MultiTaskFER()
    model = model.to(DEVICE)

    # =====================================
    # LOSS FUNCTION & OPTIMIZER
    # =====================================
    emotion_loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor.to(DEVICE))
    feature_loss_fn = MultiLabelFocalLoss(alpha=0.25, gamma=2.0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )

    # =====================================
    # LEARNING RATE SCHEDULER & EARLY STOPPING
    # =====================================
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    early_stopping = EarlyStopping(patience=5)
    
    scaler = torch.cuda.amp.GradScaler()
    
    start_epoch = 0
    best_score = 0
    
    # Định nghĩa chính xác đường dẫn của cả 2 file checkpoint
    last_checkpoint_path = os.path.join(SAVE_DIR, "last_model.pth")
    best_checkpoint_path = os.path.join(SAVE_DIR, "best_model.pth")

    # 🛠️ SỬA LOGIC KHÔI PHỤC CHECKPOINT TOÀN CỤC (GLOBAL RESUME LOGIC)
    if os.path.exists(last_checkpoint_path):
        print(f"Tim thay checkpoint dung de khoi phuc tai: {last_checkpoint_path}")
        print("Dang tien hanh khoi phuc toan bo trang thai de tiep tuc huan luyen...")
        
        # 1. Khôi phục trọng số và trạng thái optimizer từ epoch gần nhất
        checkpoint = load_checkpoint(model, optimizer, last_checkpoint_path, DEVICE)
        start_epoch = checkpoint["epoch"]
        
        # 2. Truy xuất lại kỷ lục điểm cao nhất lịch sử từ best_model.pth nếu có
        if os.path.exists(best_checkpoint_path):
            best_checkpoint = torch.load(best_checkpoint_path, map_location=DEVICE)
            best_score = (best_checkpoint.get("emotion_f1", 0) + best_checkpoint.get("feature_f1", 0)) / 2
            print(f"-> Da tim thay file lich su! Khoi phuc Global Best Score toan cuc: {best_score:.4f}")
        else:
            best_score = (checkpoint["emotion_f1"] + checkpoint["feature_f1"]) / 2
            print(f"-> Khong tim thay best_model.pth, khoi tao best_score tam thoi: {best_score:.4f}")
            
        print(f"Khoi phuc thanh cong! Tiep tuc tu Epoch: {start_epoch + 1}\n")

    # =====================================
    # TRAINING PIPELINE LOOP
    # =====================================
    model.freeze_backbone()

    if start_epoch >= 5:
        model.unfreeze_last_blocks(num_blocks=3)
        print(f"\n[Khởi động lại từ Epoch {start_epoch+1}] Đã tự động mở khóa 3 block cuối của MobileNetV3.")
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable Params: {trainable_params:,} / Total Params: {total_params:,}\n")

    for epoch in range(start_epoch, EPOCHS):
        model.train()

        if epoch == 5:
            model.unfreeze_last_blocks(num_blocks=3)
            print(f"\n[Epoch {epoch+1}] Đã mở khóa 3 block cuối của MobileNetV3 để bắt đầu giai đoạn Fine-tune!")
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Trainable Params: {trainable_params:,} / Total Params: {total_params:,}\n")

        if epoch < 5:
            model.backbone.eval()

        running_loss = 0
        train_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{EPOCHS}")

        for batch in train_bar:
            images = batch["image"].to(DEVICE, non_blocking=True)
            emotions = batch["emotion"].to(DEVICE, non_blocking=True)
            features = batch["features"].float().to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=DEVICE):
                outputs = model(images)
                emotion_logits = outputs["emotion_logits"]
                feature_logits = outputs["feature_logits"]

                emotion_loss = emotion_loss_fn(emotion_logits, emotions)
                feature_loss = feature_loss_fn(feature_logits, features)

                total_loss = emotion_loss + 0.6 * feature_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += total_loss.item()
            train_bar.set_postfix(loss=f"{total_loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # =====================================
        # VALIDATION PHASE
        # =====================================
        val_bar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{EPOCHS}", leave=True)
        emotion_result, feature_result = validate(model, val_bar, DEVICE)

        val_emotion_f1 = emotion_result["f1"]
        val_feature_f1 = feature_result["f1"]

        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss     : {avg_train_loss:.4f}")
        print(f"Val Emotion F1 : {val_emotion_f1:.4f}")
        print(f"Val Feature F1 : {val_feature_f1:.4f}")

        print_epoch_result(epoch + 1, avg_train_loss, val_emotion_f1, val_feature_f1)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Toc do hoc hien tai (Learning Rate): {current_lr}")

        current_score = (val_emotion_f1 + val_feature_f1) / 2

        # =====================================
        # PERFORMANCE CONTROL (LR & EARLY STOPPING)
        # =====================================
        scheduler.step(current_score)

        # Luôn lưu trạng thái mới nhất của epoch hiện tại vào last_model.pth
        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            val_emotion_f1,
            val_feature_f1,
            last_checkpoint_path,
        )

        # So sánh chuẩn xác với Global Best Score lịch sử đã khôi phục thành công
        if current_score > best_score:
            best_score = current_score
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_emotion_f1,
                val_feature_f1,
                best_checkpoint_path,
            )
            print(f"👉 CHÚC MỪNG: Phá vỡ kỷ lục toàn cục! Da cap nhat va luu mo hinh tot nhat moi! Score: {best_score:.4f}")
        else:
            print(f"🔒 Giữ nguyên mô hình tốt nhất cũ. Kỷ lục lịch sử vẫn là: {best_score:.4f}")

        if early_stopping.step(current_score):
            print(
                "\nKich hoat Early Stopping! Mo hinh dung huan luyen som de bao toan diem toi uu, tranh Overfitting."
            )
            break

    print("\nPipeline Phase 4 da hoan thanh huan luyen xuat sac!")


if __name__ == "__main__":
    main()