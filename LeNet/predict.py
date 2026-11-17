#!/usr/bin/env python3
"""Offline prediction and reusable LeNet-5 policy API. Does not control motors."""

import argparse
import json

import numpy as np
import torch

try:
    from .dataset import read_npz
    from .model import ACTION_NAMES, STATE_SHAPE, load_model
except ImportError:
    from dataset import read_npz
    from model import ACTION_NAMES, STATE_SHAPE, load_model


class LeNetPolicy:
    """Inference wrapper for a trained LeNet5 checkpoint."""

    def __init__(self, checkpoint, device="cpu"):
        self.device = device
        self.model = load_model(checkpoint, device)

    def predict(self, state):
        """Predict driving action from oldest->newest RGB uint8 (5, 3, 32, 32)."""
        state = np.asarray(state)
        if state.shape != STATE_SHAPE or state.dtype != np.uint8:
            raise ValueError("입력은 RGB uint8 (5, 3, 32, 32) 배열이어야 합니다.")
        tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).to(
            device=self.device, dtype=torch.float32).div_(255.0)
        with torch.inference_mode():
            probabilities = self.model(tensor).softmax(dim=1)[0].cpu().tolist()
        action = int(np.argmax(probabilities))
        return {
            "action": action,
            "name": ACTION_NAMES[action],
            "confidence": probabilities[action],
            "probabilities": probabilities,
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--npz", required=True)
    p.add_argument("--index", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(2)
    states, actions = read_npz(args.npz)
    if not 0 <= args.index < len(states):
        p.error(f"index는 0 이상 {len(states)} 미만이어야 합니다.")
    result = LeNetPolicy(args.checkpoint).predict(states[args.index])
    result["label"] = int(actions[args.index])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
