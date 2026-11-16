#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
특정 NPZ 파일 지정 시각화 및 변환 도구
경로: /home/pi/Pirrari/collector/export_npz.py

[사용법]
1. 명령어 뒤에 파일명을 지정하여 실행:
   python3 export_npz.py data/20261109_144910_000.npz
   (또는 파일명만: python3 export_npz.py 20261109_144910_000.npz)

2. 인자 없이 실행 시:
   python3 export_npz.py
   -> 발견된 파일 목록이 번호로 표시되며, 원하는 번호를 입력하여 선택할 수 있습니다.
"""

import os
import sys
import glob
import base64
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "dataset")
EXPORTS_BASE_DIR = os.path.join(BASE_DIR, "exports")

ACTION_NAMES = {
    0: ("전진 (Forward)", (0, 200, 0)),
    1: ("우회전 (Turn Right)", (255, 120, 0)),
    2: ("좌회전 (Turn Left)", (0, 215, 255)),
    3: ("정지 (Stop)", (0, 0, 220))
}

def choose_target_file():
    """인자가 있으면 인자 파싱, 없으면 대화형 번호 선택"""
    npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))

    # 1. 커맨드라인 인자로 전달된 경우
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        # 전체 경로 확인
        if os.path.isfile(arg):
            return arg
        # data/ 폴더 안의 파일명 확인
        cand = os.path.join(DATA_DIR, arg)
        if os.path.isfile(cand):
            return cand
        cand_ext = os.path.join(DATA_DIR, arg + ".npz")
        if os.path.isfile(cand_ext):
            return cand_ext

        print(f"❌ 지정한 파일을 찾을 수 없습니다: {arg}")
        print("data/ 폴더에 존재하는 파일명을 확인해주세요.\n")

    # 2. 인자가 없거나 잘못된 경우: 번호 목록 표시
    if not npz_files:
        print("❌ data/ 폴더에 .npz 파일이 없습니다.")
        sys.exit(1)

    print("=" * 60)
    print(" 📁 변환할 .npz 파일을 선택하세요:")
    print("=" * 60)
    for idx, f in enumerate(npz_files, 1):
        try:
            d = np.load(f)
            cnt = len(d["actions"])
        except Exception:
            cnt = "?"
        fname = os.path.basename(f)
        print(f"  [{idx:>2}] {fname:<30} ({cnt}개 샘플)")

    print(f"  [ A] 전체 파일 모두 합쳐서 변환")
    print("=" * 60)

    try:
        choice = input("👉 번호를 입력하세요 (예: 1): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n취소되었습니다.")
        sys.exit(0)

    if choice.lower() == 'a':
        return npz_files  # 전체 리스트 반환

    try:
        num = int(choice)
        if 1 <= num <= len(npz_files):
            return npz_files[num - 1]
    except ValueError:
        pass

    print("잘못된 입력입니다. 첫 번째 파일로 진행합니다.")
    return npz_files[0]

def export_video(states, actions, out_path, fps=10, scale=10):
    N = len(actions)
    h, w = 32 * scale, 32 * scale
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h + 60))

    for i in range(N):
        img_ch_first = states[i, 4]  # 최근 프레임 F5
        img_bgr = cv2.cvtColor(img_ch_first.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_NEAREST)

        panel = np.zeros((60, w, 3), dtype=np.uint8)
        act_code = int(actions[i])
        act_name, color = ACTION_NAMES.get(act_code, (f"Action {act_code}", (200, 200, 200)))

        cv2.putText(panel, f"Sample: {i+1}/{N}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(panel, f"ACTION: {act_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        combined = np.vstack([img_resized, panel])
        out.write(combined)

    out.release()
    print(f"  🎬 [비디오 생성] {out_path}")

def export_images(states, actions, out_dir, max_count=50):
    os.makedirs(out_dir, exist_ok=True)
    N = min(len(actions), max_count)

    for i in range(N):
        frames = []
        for f in range(5):
            img_bgr = cv2.cvtColor(states[i, f].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            img_big = cv2.resize(img_bgr, (120, 120), interpolation=cv2.INTER_NEAREST)
            cv2.putText(img_big, f"F{f+1}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            frames.append(img_big)

        strip = np.hstack(frames)
        act_code = int(actions[i])
        act_name = ACTION_NAMES.get(act_code, (f"act{act_code}", None))[0].split()[0]
        fname = f"sample_{i+1:04d}_act_{act_code}_{act_name}.png"
        cv2.imwrite(os.path.join(out_dir, fname), strip)

    print(f"  🖼️  [이미지 추출] {out_dir}/ (총 {N}개)")

def export_html(states, actions, out_html_path, title_name, max_items=100):
    N = min(len(actions), max_items)
    rows_html = []

    for i in range(N):
        frames = []
        for f in range(5):
            img_bgr = cv2.cvtColor(states[i, f].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            img_big = cv2.resize(img_bgr, (100, 100), interpolation=cv2.INTER_NEAREST)
            cv2.putText(img_big, f"F{f+1}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            frames.append(img_big)
        combined = np.hstack(frames)
        _, buf = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_str = base64.b64encode(buf).decode('utf-8')

        act_code = int(actions[i])
        act_name, _ = ACTION_NAMES.get(act_code, (f"Action {act_code}", None))
        badge_color = {0: "#28a745", 1: "#007bff", 2: "#ffc107", 3: "#dc3545"}.get(act_code, "#6c757d")
        text_color = "#000" if act_code == 2 else "#fff"

        row = f"""
        <tr>
            <td style="text-align:center; font-weight:bold;">#{i+1}</td>
            <td style="text-align:center;">
                <span style="background:{badge_color}; color:{text_color}; padding:4px 10px; border-radius:12px; font-weight:bold; font-size:13px;">
                    {act_name}
                </span>
            </td>
            <td>
                <img src="data:image/jpeg;base64,{b64_str}" style="border-radius:6px; box-shadow:0 1px 4px rgba(0,0,0,0.2);" />
            </td>
        </tr>
        """
        rows_html.append(row)

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>{title_name} - 데이터셋 뷰어</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8f9fa; margin: 0; padding: 20px; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
        h1 {{ margin-top: 0; color: #333; font-size: 24px; }}
        .stats {{ background: #e9ecef; padding: 12px 18px; border-radius: 8px; margin-bottom: 20px; font-size: 14px; line-height: 1.6; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; border-bottom: 1px solid #dee2e6; }}
        th {{ background: #f1f3f5; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 {title_name} 뷰어</h1>
        <div class="stats">
            <b>• 대상 데이터:</b> {title_name}<br>
            <b>• 총 샘플 수:</b> {len(actions)}개 (HTML 표시: 상위 {N}개)<br>
            <b>• 사진 열:</b> F1(가장 오래된 과거) ➔ F5(현재 프레임) 5장 연속 사진
        </div>
        <table>
            <thead>
                <tr>
                    <th style="width: 80px;">번호</th>
                    <th style="width: 180px;">조작 키 (Action)</th>
                    <th>5장 연속 프레임 (F1 ➔ F5)</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  🌐 [HTML 뷰어 생성] {out_html_path}")

def main():
    target = choose_target_file()

    if isinstance(target, list):
        tag_name = "ALL_COMBINED"
        all_states = [np.load(f)["states"] for f in target]
        all_actions = [np.load(f)["actions"] for f in target]
        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)
    else:
        tag_name = os.path.splitext(os.path.basename(target))[0]
        data = np.load(target)
        states = data["states"]
        actions = data["actions"]

    # 각 파일별 전용 출력 폴더 생성 (예: exports/20261109_144910_000/)
    out_dir = os.path.join(EXPORTS_BASE_DIR, tag_name)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f" 🚀 변환 대상: {tag_name} (총 {len(actions)}개 샘플)")
    print(f" 📂 저장 위치: {out_dir}/")
    print("=" * 60)

    # 1. MP4 주행 동영상
    video_path = os.path.join(out_dir, "drive_replay.mp4")
    export_video(states, actions, video_path)

    # 2. HTML 뷰어
    html_path = os.path.join(out_dir, "data_viewer.html")
    export_html(states, actions, html_path, tag_name)

    # 3. 개별 이미지
    img_dir = os.path.join(out_dir, "images")
    export_images(states, actions, img_dir, max_count=30)

    print("\n" + "=" * 60)
    print(f"🎉 변환 완료!")
    print(f"  • 비디오 : {video_path}")
    print(f"  • 웹뷰어 : {html_path}")
    print(f"  • 이미지 : {img_dir}/")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
