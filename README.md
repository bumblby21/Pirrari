# Pirrari

Raspbot 기반 키보드 주행 제어 및 Imitation Learning 데이터셋 수집 프로젝트입니다.

## 폴더 구조

```text
Pirrari/
├── collector/
│   ├── keyboard_control.py   # 키보드 주행 및 데이터 수집 메인 스크립트
│   ├── check_data.py         # 수집된 데이터셋 통계 및 미리보기 생성
│   ├── export_npz.py         # .npz 데이터셋을 이미지/영상/HTML로 내보내기
│   ├── collect.py            # UDP 원격 수집 스크립트
│   └── verify_wifi_safety.py # 네트워크 단절 안전 정지 검증 스크립트
└── dataset/                  # 수집된 .npz 파일 저장 경로
```

## 바로 실행하기

### 1. 주행 및 데이터 수집 실행

```bash
cd /home/pi/Pirrari/collector
python3 keyboard_control.py
```

### 2. 수집된 데이터셋 확인

`dataset/` 폴더에 저장된 샘플 개수, 액션별 비율 확인 및 5프레임 미리보기 이미지(`preview.png`)를 생성합니다.

```bash
cd /home/pi/Pirrari/collector
python3 check_data.py
```

### 3. 데이터셋 영상/이미지 내보내기

```bash
cd /home/pi/Pirrari/collector
python3 export_npz.py
```

특정 파일만 내보내려면 파일명을 인자로 전달합니다.

```bash
python3 export_npz.py 20261116_140000_000.npz
```

---

## 조작 키 안내 (keyboard_control.py)

키를 계속 누르고 있을 필요 없이 한 번 입력하면 다음 명령 전까지 주행 상태가 유지됩니다.

| 키 | 동작 | 비고 |
| --- | --- | --- |
| `W` 또는 위 방향키 | 전진 | Action 0 |
| `D` 또는 오른쪽 방향키 | 우회전 | Action 1 |
| `A` 또는 왼쪽 방향키 | 좌회전 | Action 2 |
| `S` 또는 아래 방향키 | 후진 | 데이터셋 저장 자동 제외 |
| `Space` 또는 `X` | 즉시 정지 | 데이터셋 저장 자동 제외 |
| `R` | 녹화 시작 / 일시 정지 토글 | 정지 및 후진 상태는 자동 제외 |
| `+` / `-` | 주행 및 회전 속도 조절 | 기본 속도 28 / 회전 25, 2단위 증감 |
| `Q` 또는 `ESC` | 프로그램 종료 | 미저장 데이터 존재 시 `y/n` 저장 여부 선택 |

---

## 데이터 수집 및 안전 기능 요약

- **주행 데이터만 자동 필터링 저장**: `R`로 녹화를 켜둔 상태에서 정지(`Space`/`X`)하거나 후진(`S`)해도 녹화 모드는 유지되며, 실제 학습용 데이터(`dataset/*.npz`)와 `5000개` 목표 카운트에는 전진(`0`), 우회전(`1`), 좌회전(`2`)만 포함됩니다.
- **중간 자동 저장**: 주행 데이터가 `500개` 모일 때마다 `dataset/YYYYMMDD_HHMMSS_NNN.npz` 형식으로 자동 압축 저장됩니다.
- **종료 시 저장 선택**: `Q` 또는 `ESC`로 정상 종료할 때 미저장 데이터가 남아 있으면 `y/n` 입력을 통해 저장 또는 폐기를 선택할 수 있습니다.
- **비상 안전 정지 및 자동 저장**: 와이파이 단절, 유선 랜선 분리, 터미널 창 닫기(`SIGHUP`/`EOF`), `Ctrl+C` 등 갑작스런 연결 단절 시 0.2~0.4초 이내에 모터를 0으로 정지하고 미저장 데이터를 자동으로 디스크에 저장합니다.

---

## GitHub 사용 방법 (다운로드 및 업로드)

### 1. 처음 내려받기 (새 PC / 새 환경)

```bash
git clone https://github.com/bumblby21/Pirrari.git
cd Pirrari
```

### 2. 최신 코드 내려받기 (이미 폴더가 있을 때)

작업 시작 전 다른 팀원이 올린 최신 변경 사항을 반영합니다.

```bash
cd /home/pi/Pirrari
git pull origin main
```

### 3. 수정 후 GitHub에 올리기 (본인 계정으로 커밋 남기기)

라즈베리파이 공용 환경에서 코드를 수정한 뒤 본인 이름으로 커밋 기록을 남기려면 `--author` 옵션을 사용합니다.

```bash
cd /home/pi/Pirrari
git status
git add .
git commit --author="본인이름 <본인GitHub이메일>" -m "작업 내용"
git push origin main
```

- 다른 팀원이 먼저 코드를 올렸을 때는 `git pull origin main`으로 먼저 최신 내용을 받은 뒤 `git push origin main`을 실행합니다.
