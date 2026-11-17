"""LeNet-5 architecture adapted for 5 stacked 32x32 RGB frames."""

import torch
from torch import nn

ACTION_NAMES = ["forward", "right", "left"]
STATE_SHAPE = (5, 3, 32, 32)
ARCH_NAME = "LeNet5"


class LeNet5(nn.Module):
    """Classic LeNet-5 (C1 -> S2 -> C3 -> S4 -> C5 -> F6 -> Output) for [B, 5, 3, 32, 32]."""

    def __init__(self, num_classes=len(ACTION_NAMES), dropout=0.2):
        super().__init__()
        in_channels = STATE_SHAPE[0] * STATE_SHAPE[1]  # 15 channels (5 RGB frames)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),   # (15, 32, 32) -> (6, 28, 28)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),      # (6, 28, 28)  -> (6, 14, 14)
            nn.Conv2d(6, 16, kernel_size=5),            # (6, 14, 14)  -> (16, 10, 10)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),      # (16, 10, 10) -> (16, 5, 5)
            nn.Conv2d(16, 120, kernel_size=5),          # (16, 5, 5)   -> (120, 1, 1)
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(84, num_classes),
        )

    def forward(self, states):
        return self.classifier(self.features(states.flatten(1, 2)))


def load_model(checkpoint_path, device="cpu"):
    """Load a trained LeNet5 checkpoint and set it to eval mode."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("지원하지 않는 체크포인트 형식입니다.")
    if checkpoint.get("arch") != ARCH_NAME:
        raise ValueError(f"LeNet5 체크포인트가 아닙니다: {checkpoint.get('arch')}")
    if checkpoint.get("action_names") != ACTION_NAMES:
        raise ValueError("체크포인트의 액션 순서가 다릅니다.")
    if checkpoint.get("state_shape") != list(STATE_SHAPE):
        raise ValueError("체크포인트의 입력 크기가 다릅니다.")
    model = LeNet5().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
