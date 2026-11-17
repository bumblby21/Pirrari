# LeNet-5 자율주행 학습 및 실행 가이드

`dataset/1117_basic_2000`과 `dataset/1117_midtwobox_500` 데이터셋을 **LeNet-5** 모델로 학습하고 Raspbot 자율주행까지 실행하는 패키지입니다.

## 모델 구조 (`LeNet5`)
- 입력: `(5, 3, 32, 32)` 최근 5프레임 RGB 이미지 → 채널 결합 `(15, 32, 32)`
- `Conv2d(15→6, 5x5)` → `ReLU` → `AvgPool2d(2x2)`
- `Conv2d(6→16, 5x5)` → `ReLU` → `AvgPool2d(2x2)`
- `Conv2d(16→120, 5x5)` → `ReLU` → `Flatten`
- `Linear(120→84)` → `ReLU` → `Dropout(0.2)` → `Linear(84→3)` (`forward`, `right`, `left`)

## 사용 방법

### 1. 학습만 실행
```bash
cd /home/pi/Pirrari
/home/pi/yolov5-lite/bin/python LeNet/run.py --train-only
```
기본적으로 `dataset/1117_basic_2000`과 `dataset/1117_midtwobox_500`을 함께 불러와 학습하며,
최적 가중치는 `LeNet/runs/<시간>/best.pt` 및 `LeNet/models/best.pt`에 저장됩니다.

### 2. 학습 후 카메라 예측 테스트 (모터 미사용)
```bash
cd /home/pi/Pirrari
/home/pi/yolov5-lite/bin/python LeNet/run.py
```

### 3. 학습 후 바로 실제 자율주행 (`--drive`)
```bash
cd /home/pi/Pirrari
/home/pi/yolov5-lite/bin/python LeNet/run.py --drive
```

### 4. 이미 학습된 모델(`LeNet/models/best.pt`)로 바로 자율주행
```bash
cd /home/pi/Pirrari
/home/pi/yolov5-lite/bin/python LeNet/drive.py --drive
```

- 정지/종료: `Space`, `X`, `Q`, `ESC`, `Ctrl+C`
- 속도/회전 조절: `--speed 28 --turn 25`
- 주요 안전장치(`MotorGuard`, 카메라/추론 타임아웃, SSH 연결 감시)는 기존 `autonomous/` 모듈을 그대로 사용합니다.
