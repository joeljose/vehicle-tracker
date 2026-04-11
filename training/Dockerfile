# Training pipeline image
# CUDA runtime base + Python + ffmpeg. Used for frame extraction, dedup,
# YOLOv8 auto-labeling, and (later) fine-tuning.

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
      python3.12 python3-pip \
      ffmpeg \
      libgl1 libglib2.0-0 \
      # X11 libs for opencv-python's bundled Qt5 (needed by build_benchmark.py
      # editor when forwarded over X11 from the host).
      libsm6 libice6 libxext6 libxrender1 libxkbcommon0 \
      libfontconfig1 libdbus-1-3 \
      > /dev/null 2>&1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

WORKDIR /app

CMD ["tail", "-f", "/dev/null"]
