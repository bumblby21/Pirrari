#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로컬 dataset/ 폴더의 모든 .npz 파일을 지정된 Google Drive 폴더로 동기화하는 스크립트
경로: /home/pi/Pirrari/collector/sync_gdrive.py
대상 폴더: https://drive.google.com/drive/folders/1rtkqgbBGJexCRJi4LR0AorAuduAUPmMf
"""
import os
import subprocess
import sys

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
GDRIVE_REMOTE = "gdrive:"
GDRIVE_FOLDER_ID = "1rtkqgbBGJexCRJi4LR0AorAuduAUPmMf"


def main():
    res = subprocess.run(["rclone", "listremotes"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if "gdrive:" not in res.stdout:
        print("[안내] 아직 rclone에 'gdrive' 리모트가 설정되지 않았습니다.")
        print("먼저 'rclone config'를 실행하여 Google Drive 리모트 이름 'gdrive'를 등록해 주세요.")
        sys.exit(1)

    print(f"로컬 폴더({DATASET_DIR}) -> 구글 드라이브 폴더({GDRIVE_FOLDER_ID}) 업로드 시작...")
    cmd = [
        "rclone", "copy", DATASET_DIR, GDRIVE_REMOTE,
        "--drive-root-folder-id", GDRIVE_FOLDER_ID,
        "--include", "*.npz",
        "-P"
    ]
    proc = subprocess.run(cmd)
    if proc.returncode == 0:
        print("구글 드라이브 동기화가 완료되었습니다.")
    else:
        print(f"동기화 중 오류가 발생했습니다 (exit code: {proc.returncode}).")
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
