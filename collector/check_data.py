#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터 검증 및 시각화 도구
경로: /home/pi/Pirrari/collector/check_data.py

[기능]
1. data/ 폴더 내의 모든 .npz 파일 자동 검색 및 총 수집 개수 확인
2. 액션별(전진/우회전/좌회전/정지) 데이터 분포 통계 출력
3. 실제 저장된 5장 연속 프레임을 이어붙여 'preview.png'로 저장 (눈으로 사진 확인 가능)
"""

import os
import glob
import numpy as np
import cv2

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
ACTION_NAMES = {0: "전진 (Forward)", 1: "우회전 (Turn Right)", 2: "좌회전 (Turn Left)", 3: "정지 (Stop)"}

def main():
    print("=" * 60)
    print("           수집된 Imitation Learning 데이터 점검")
    print("=" * 60)

    npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if not npz_files:
        print("❌ data/ 폴더에 저장된 .npz 파일이 없습니다.")
        return

    total_samples = 0
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    latest_state = None
    latest_action = None

    print(f"📁 발견된 데이터 파일 목록 ({len(npz_files)}개):")
    for f in npz_files:
        try:
            data = np.load(f)
            states = data["states"]
            actions = data["actions"]
            n = len(actions)
            total_samples += n
            file_name = os.path.basename(f)
            print(f"  - {file_name:<25} : {n:>4}개 샘플 | states: {states.shape}")

            for a in actions:
                if a in action_counts:
                    action_counts[a] += 1

            if n > 0:
                latest_state = states[-1]     # (5, 3, 32, 32)
                latest_action = actions[-1]
        except Exception as e:
            print(f"  - {os.path.basename(f)} 읽기 오류: {e}")

    print("-" * 60)
    print(f"📊 [전체 통계 요약]")
    print(f"  • 총 수집 샘플 수 : {total_samples} / 5000 개 (달성률: {total_samples/5000*100:.1f}%)")
    print(f"  • 액션별 분포:")
    for code, name in ACTION_NAMES.items():
        cnt = action_counts[code]
        pct = (cnt / total_samples * 100) if total_samples > 0 else 0
        print(f"      [{code}] {name:<18} : {cnt:>4}개 ({pct:>5.1f}%)")

    # 최근 프레임 5장을 가로로 이어붙여 preview.png로 저장
    if latest_state is not None:
        # latest_state: (5, 3, 32, 32) -> RGB to BGR for cv2
        frames = []
        for i in range(5):
            img_ch_first = latest_state[i] # (3, 32, 32)
            img_hwc = img_ch_first.transpose(1, 2, 0) # (32, 32, 3) RGB
            img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
            # 눈에 잘 보이도록 확대 (32x32 -> 160x160)
            img_large = cv2.resize(img_bgr, (160, 160), interpolation=cv2.INTER_NEAREST)
            # 프레임 번호 텍스트 추가
            cv2.putText(img_large, f"F{i+1}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            frames.append(img_large)

        # 5장을 가로로 결합 (가로 800, 세로 160)
        combined_img = np.hstack(frames)
        preview_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview.png")
        cv2.imwrite(preview_path, combined_img)

        print("-" * 60)
        print(f"🖼️ [사진 미리보기 생성 완료]")
        print(f"  • 저장된 5장 연속 프레임이 이미지로 추출되었습니다: preview.png")
        print(f"  • 마지막 샘플 액션: {ACTION_NAMES.get(latest_action, latest_action)}")
        print(f"  • 파일 경로: {preview_path}")

    print("=" * 60)

if __name__ == "__main__":
    main()
