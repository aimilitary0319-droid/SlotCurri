# SlotCurri Docker 환경

CVPR 2026 **SlotCurri** (Reconstruction-Guided Slot Curriculum) 학습 및 실험을 위한 컨테이너 구성입니다.

## 구성 요약

| 항목 | 값 |
|---|---|
| Base image | `pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel` |
| Python | 3.10 |
| 의존성 관리 | Poetry 1.8.3 (`pyproject.toml`, `poetry.lock`) |
| 포함 extras | `tensorflow`, `coco`, `notebook` 전부 |
| 추가 pip 패키지 | `pytorch_msssim`, `kornia`, `lpips`, `gdown`, `opencv-python-headless`, `tensorboard`, `wandb` |
| GPU 런타임 | NVIDIA (CDI / nvidia-container-toolkit 둘 다 호환) |
| 사용자 | non-root (`slotcurri`, 기본 UID/GID 1000) |

호스트의 GPU 드라이버는 CUDA 11.6 이상을 지원해야 합니다(현 호스트: 드라이버 580 / CUDA 13.0 → OK).

## 디렉터리 가정

```
/mnt/ssd2/hmlee/
├── SlotCurri/        ← 이 저장소 (컨테이너 안에서 /workspace/SlotCurri 로 마운트)
│   └── docker/       ← 이 폴더
└── dataset/          ← 데이터셋 루트 (컨테이너 안에서 /workspace/dataset 으로 마운트)
    ├── ytvis2021_resized/
    ├── movi_c/
    └── movi_e/
```

데이터셋 위치를 바꾸고 싶다면 `docker-compose.yml` 의 `volumes:` 항목을 수정하세요.

## 1. 초기 설정

```bash
cd /mnt/ssd2/hmlee/SlotCurri/docker

# 호스트 UID/GID 와 일치시키면 마운트한 파일의 권한 문제가 사라집니다.
cp .env.example .env
sed -i "s/^USER_UID=.*/USER_UID=$(id -u)/" .env
sed -i "s/^USER_GID=.*/USER_GID=$(id -g)/" .env
```

## 2. 이미지 빌드

```bash
cd /mnt/ssd2/hmlee/SlotCurri/docker
docker compose build
```

> 처음 빌드는 의존성 다운로드 때문에 10~20분 정도 걸릴 수 있습니다.

## 3. 컨테이너 기동 & 진입

```bash
docker compose up -d
docker compose exec slotcurri bash
```

컨테이너 내부에서 GPU 확인:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

## 4. 학습 실행

`pyproject.toml` 의 의존성이 시스템 Python 에 바로 설치되어 있으므로 `poetry run` 없이도 호출됩니다.
README 의 원본 명령을 그대로 쓰고 싶다면 `poetry run` 도 사용 가능합니다.

```bash
# 컨테이너 내부에서
cd /workspace/SlotCurri

# YouTube-VIS 2021
python -m slotcurri.train --run-eval-after-training configs/slotcurri/ytvis2021.yaml

# MOVi-E
python -m slotcurri.train --run-eval-after-training configs/slotcurri/movi_e.yaml

# MOVi-C
python -m slotcurri.train --run-eval-after-training configs/slotcurri/movi_c.yaml
```

원본 README 의 `<root_data_dir>` 가 컨테이너 안에서는 `/workspace/dataset` 입니다.

## 5. 자주 쓰는 명령

```bash
# 컨테이너 재시작
docker compose restart

# 컨테이너 정지 / 제거
docker compose down

# 로그 확인 (백그라운드 학습 시)
docker compose logs -f slotcurri

# 멀티 셸 열기
docker compose exec slotcurri bash

# 일회성 명령 실행
docker compose exec slotcurri python -m slotcurri.eval ...

# Jupyter (호스트 8888 → 컨테이너 8888)
docker compose exec slotcurri \
    jupyter notebook --ip=0.0.0.0 --no-browser --NotebookApp.token=''

# TensorBoard (호스트 6006)
docker compose exec slotcurri tensorboard --logdir runs --host 0.0.0.0
```

## 6. 데이터셋 준비 예시

컨테이너 안에서:

```bash
cd /workspace/SlotCurri/data

# MOVi-C
python save_movi.py --level c --split train      --maxcount 32 --only-video /workspace/dataset/movi_c
python save_movi.py --level c --split validation --maxcount 32              /workspace/dataset/movi_c

# YouTube-VIS 2021
python save_ytvis2021.py --split train      --maxcount 32 --only-videos --resize --out-path /workspace/dataset/ytvis2021_resized
python save_ytvis2021.py --split validation --maxcount 10               --resize --out-path /workspace/dataset/ytvis2021_resized
```

## 7. 트러블슈팅

- **`could not select device driver "" with capabilities: [[gpu]]`**: `nvidia-container-toolkit` 가 설치되어 있는지 확인하세요. CDI 사용 환경이라면 `docker compose.yml` 의 `deploy.resources` 대신 `devices: ["nvidia.com/gpu=all"]` 로 바꿔도 됩니다.
- **DataLoader 가 `Bus error` 로 죽음**: 보통 shared memory 부족. 이미 `shm_size: 32gb` 로 설정되어 있지만 더 늘리려면 compose 파일에서 조정하세요.
- **`Permission denied` 로 파일 못 씀**: `.env` 의 `USER_UID/USER_GID` 가 호스트와 일치하는지 확인 후 `docker compose build --no-cache`.
- **CUDA OOM**: 학습 yaml 의 batch size / num workers / image size 를 조정하거나, `NVIDIA_VISIBLE_DEVICES` 로 GPU 개수를 늘려서 멀티 GPU 학습.
- **`poetry.lock` 충돌**: `pyproject.toml` 을 수정했다면 호스트에서 `poetry lock --no-update` 후 다시 빌드하거나, 컨테이너 안에서 `poetry install` 만 다시 실행.

## 8. 파일 구조

```
docker/
├── Dockerfile          # 이미지 정의
├── docker-compose.yml  # 컨테이너 실행 정의 (GPU, 마운트, 포트)
├── .dockerignore       # 빌드 컨텍스트 제외 목록
├── .env.example        # 사용자 UID/GID, 포트 등 환경 변수 템플릿
└── README.md           # 이 문서
```
