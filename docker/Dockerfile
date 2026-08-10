# =============================================================================
# SlotCurri - Training & Experimentation Container
#
# Base : PyTorch 1.13.1 + CUDA 11.6 + cuDNN 8 + Python 3.10
#        (works on hosts with newer NVIDIA drivers thanks to CUDA forward-compat)
# =============================================================================
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_HOME=/opt/poetry \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PATH="/opt/poetry/bin:${PATH}" \
    TZ=Asia/Seoul

# ---- System packages ---------------------------------------------------------
# - git/curl/ca-certificates : poetry, gdown 등을 위한 기본 도구
# - build-essential          : pycocotools, lpips 등 C 확장 빌드
# - ffmpeg/libsm6/libxext6   : av / moviepy / OpenCV 사용 위한 코덱 라이브러리
# - libgl1                   : OpenCV / torchvision 일부 기능 의존
# Docker 자체의 --init (compose 에서는 init:true) 가 PID 1 시그널 처리.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        ca-certificates \
        build-essential \
        pkg-config \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        libglib2.0-0 \
        sudo \
        vim \
        tmux \
        htop \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

# ---- Poetry ------------------------------------------------------------------
RUN curl -sSL https://install.python-poetry.org | python - --version ${POETRY_VERSION} \
    && poetry --version

# ---- Project dependencies ----------------------------------------------------
WORKDIR /workspace/SlotCurri

# pyproject.toml / poetry.lock 만 먼저 복사하여 의존성 캐시 활용
COPY pyproject.toml poetry.lock ./

# README 가 [tool.poetry] readme 필드로 참조되므로 빈 placeholder 생성
RUN touch README.md

# 모든 extras (tensorflow, coco, notebook) 포함하여 설치
RUN poetry install --no-root --all-extras

# 논문에서 추가로 요구하는 의존성들.
# 중요: torch 1.13.1 / numpy 1.x 와의 호환을 유지하기 위해 버전을 pin 한다.
# kornia 0.7.x+ 는 torch 2.x 를 의존성으로 끌어오므로 0.6.12 사용.
# wandb 0.20+ 는 Go 소스 빌드가 필요하므로 마지막 wheel 배포 버전을 pin.
RUN pip install --no-cache-dir \
        "pytorch-msssim==1.0.0" \
        "kornia==0.6.12" \
        "lpips==0.1.4" \
        "gdown==5.2.0" \
        "opencv-python-headless==4.10.0.84" \
        "tensorboard>=2.11,<2.16" \
        "wandb<0.20" \
    && python -c "import torch, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__); assert torch.__version__.startswith('1.13'), torch.__version__; assert numpy.__version__.startswith('1.'), numpy.__version__"

# ---- Non-root user (host UID/GID 매칭 가능) ----------------------------------
ARG USER_NAME=slotcurri
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid ${USER_GID} ${USER_NAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USER_NAME} \
    && echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p /workspace/SlotCurri /workspace/dataset \
    && chown -R ${USER_UID}:${USER_GID} /workspace

USER ${USER_NAME}

# bash 히스토리/캐시는 컨테이너 안에서 영속 유지하기 위한 디렉터리
RUN mkdir -p /home/${USER_NAME}/.cache /home/${USER_NAME}/.config

WORKDIR /workspace/SlotCurri

CMD ["bash"]
