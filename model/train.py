#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pirrari Imitation Learning 모델 학습 스크립트 (TensorFlow / Keras)
경로: /home/pi/Pirrari/model/train.py

- dataset/*.npz 파일들을 모두 불러와 공간 위치 정보를 보존하는 경량 2D CNN(32x32x15)을 학습합니다.
- SpatialDropout2D, Dropout, L2 정규화, Label Smoothing, 밝기/대비·좌우반전 증강을 적용합니다.
- 학습 완료 후 pirrari_model.h5 저장 및 검증 정확도·혼동 행렬을 출력합니다.
"""

import os
import sys
import glob
import time
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ===== 경로 및 하이퍼파라미터 =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(MODEL_DIR, "saved_models")
KERAS_PATH = os.path.join(MODEL_DIR, "pirrari_model.h5")
TFLITE_PATH = os.path.join(MODEL_DIR, "pirrari_model.tflite")

EPOCHS = 45
BATCH_SIZE = 64
SEED = 42
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
ACTION_NAMES = {0: "전진(0)", 1: "우회전(1)", 2: "좌회전(2)"}


def load_dataset(data_dir):
    """dataset/ 및 하위 폴더 내 모든 .npz 파일을 로드하고 유효한 액션(0, 1, 2)만 추출합니다."""
    target_dir = os.path.join(data_dir, sys.argv[1]) if len(sys.argv) > 1 else data_dir
    all_npz = sorted(glob.glob(os.path.join(target_dir, "**", "*.npz"), recursive=True))
    npz_files = [p for p in all_npz if "backup_test" not in os.path.relpath(p, data_dir).split(os.sep)]
    if not npz_files:
        print(f"❌ {target_dir} 에 학습할 .npz 파일이 없습니다.")
        sys.exit(1)

    all_states = []
    all_actions = []
    for path in npz_files:
        data = np.load(path)
        s = data["states"]
        a = data["actions"]
        rel_path = os.path.relpath(path, data_dir)
        print(f"  • 로드: {rel_path} ({len(a)}개)")
        all_states.append(s)
        all_actions.append(a)

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)

    valid_mask = np.isin(actions, [0, 1, 2])
    states = states[valid_mask]
    actions = actions[valid_mask]

    if len(actions) < 20:
        print(f"❌ 유효한 학습 데이터가 너무 적습니다 ({len(actions)}개). 최소 20개 이상 수집해 주세요.")
        sys.exit(1)

    return states, actions


def preprocess_states(states):
    """(N, 5, 3, 32, 32) -> (N, 32, 32, 15) float32 [0, 1] 변환"""
    n, f, c, h, w = states.shape
    x = states.transpose(0, 3, 4, 1, 2).reshape(n, h, w, f * c)
    return x.astype(np.float32) / 255.0


def augment_horizontal_flip(x_train, y_train):
    """학습 데이터 좌우 반전 증강 (우회전 1 ↔ 좌회전 2 라벨 교환)"""
    x_flip = x_train[:, :, ::-1, :].copy()
    y_flip = y_train.copy()
    y_flip[y_train == 1] = 2
    y_flip[y_train == 2] = 1

    x_aug = np.concatenate([x_train, x_flip], axis=0)
    y_aug = np.concatenate([y_train, y_flip], axis=0)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y_aug))
    return x_aug[idx], y_aug[idx]


def augment_brightness_contrast(x, y, w):
    """학습 중 배치별 약한 밝기/대비 변화(±15%)를 적용"""
    contrast = tf.random.uniform([], 0.85, 1.15)
    brightness = tf.random.uniform([], -0.10, 0.10)
    mean = tf.reduce_mean(x, axis=[0, 1], keepdims=True)
    x = (x - mean) * contrast + mean + brightness
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x, y, w


def build_model(input_shape=(32, 32, 15), num_classes=3):
    """공간 위치를 보존(Flatten)하고 Dropout + SpatialDropout2D + L2를 적용한 경량 2D CNN"""
    reg = regularizers.l2(WEIGHT_DECAY)
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(24, 3, padding="same", use_bias=False, kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),

        layers.Conv2D(48, 3, padding="same", use_bias=False, kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),

        layers.Conv2D(64, 3, padding="same", use_bias=False, kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),
        layers.SpatialDropout2D(0.15),

        # 4x4 공간 위치를 유지한 채 채널만 32개로 압축 후 Flatten
        layers.Conv2D(32, 1, use_bias=False, kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Flatten(),

        layers.Dropout(0.35),
        layers.Dense(64, activation="relu", kernel_regularizer=reg),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    return model


def export_tflite_safe(model, tflite_path):
    """Keras 모델을 TFLite로 변환 시도 (환경 버전 불일치 시 건너뜀)"""
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_bytes = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_bytes)
        print(f"   - TFLite: {tflite_path}")
    except Exception:
        pass


def main():
    tf.keras.utils.set_random_seed(SEED)
    print("=" * 60)
    print(" 🚗 Pirrari Imitation Learning 모델 학습 시작")
    print("=" * 60)

    states, actions = load_dataset(DATA_DIR)
    print(f"\n📊 총 유효 데이터: {len(actions)}개")
    for cls in (0, 1, 2):
        cnt = int(np.sum(actions == cls))
        pct = 100.0 * cnt / len(actions)
        print(f"   - {ACTION_NAMES[cls]}: {cnt}개 ({pct:.1f}%)")

    x = preprocess_states(states)

    counts = [int(np.sum(actions == c)) for c in (0, 1, 2)]
    stratify_target = actions if all(c >= 2 for c in counts) else None
    x_train, x_val, y_train, y_val = train_test_split(
        x, actions, test_size=0.15, random_state=SEED, stratify=stratify_target
    )

    x_train_aug, y_train_aug = augment_horizontal_flip(x_train, y_train)
    print(f"\n🔁 학습 데이터 (좌우반전 증강 후): {len(y_train_aug)}개 / 검증 데이터: {len(y_val)}개")

    present_classes = np.unique(y_train_aug)
    weights = compute_class_weight("balanced", classes=present_classes, y=y_train_aug)
    class_weight = {int(c): float(w) for c, w in zip(present_classes, weights)}
    print(f"⚖️  클래스 가중치: {class_weight}")

    y_train_onehot = keras.utils.to_categorical(y_train_aug, num_classes=3)
    y_val_onehot = keras.utils.to_categorical(y_val, num_classes=3)
    sample_weights = np.array([class_weight[int(c)] for c in y_train_aug], dtype=np.float32)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train_aug, y_train_onehot, sample_weights))
        .shuffle(len(y_train_aug), seed=SEED)
        .map(augment_brightness_contrast, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = build_model(input_shape=x.shape[1:], num_classes=3)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            KERAS_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", mode="max", factor=0.5, patience=5, min_lr=1e-5, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=14, restore_best_weights=True, verbose=1
        ),
    ]

    model.fit(
        train_ds,
        validation_data=(x_val, y_val_onehot),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    if os.path.exists(KERAS_PATH):
        model = keras.models.load_model(KERAS_PATH)

    val_loss, val_acc = model.evaluate(x_val, y_val_onehot, verbose=0)
    preds = np.argmax(model.predict(x_val, verbose=0), axis=1)

    print("\n" + "=" * 60)
    print(f" ✅ 검증 정확도(Val Accuracy): {val_acc * 100:.2f}% (Loss: {val_loss:.4f})")
    for cls in (0, 1, 2):
        mask = y_val == cls
        if np.any(mask):
            cls_acc = 100.0 * float(np.mean(preds[mask] == cls))
            print(f"   - {ACTION_NAMES[cls]} 정확도: {cls_acc:.1f}% ({int(np.sum(mask))}개)")

    cm = confusion_matrix(y_val, preds, labels=[0, 1, 2])
    print("\n📋 혼동 행렬 (행: 실제 정답 / 열: 모델 예측 [전진, 우회전, 좌회전]):")
    for i, row in enumerate(cm):
        print(f"   실제 {ACTION_NAMES[i]:<7} -> {row.tolist()}")

    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    archive_name = f"pirrari_{run_stamp}_acc{val_acc * 100:.1f}_n{len(actions)}.h5"
    archive_path = os.path.join(SAVED_MODELS_DIR, archive_name)
    shutil.copy2(KERAS_PATH, archive_path)

    export_tflite_safe(model, TFLITE_PATH)
    size_kb = os.path.getsize(KERAS_PATH) / 1024.0
    print(f"\n💾 모델 저장 완료:")
    print(f"   - 최신 기본 모델 : {KERAS_PATH} ({size_kb:.1f} KB)")
    print(f"   - 히스토리 보관  : {archive_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
