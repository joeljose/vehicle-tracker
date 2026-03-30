# Backend development image
# Extends the NVIDIA DeepStream base with project dependencies
# so containers are ready to use immediately after build.

FROM nvcr.io/nvidia/deepstream:8.0-gc-triton-devel

# Install ffmpeg (static binary — apt packages have missing shared libs
# in the DeepStream container's stripped-down Ubuntu)
RUN apt-get update -qq && \
    apt-get install -y -qq wget xz-utils > /dev/null 2>&1 && \
    wget -q -O /tmp/ffmpeg.tar.xz \
      https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar xf /tmp/ffmpeg.tar.xz -C /tmp && \
    cp /tmp/ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ && \
    cp /tmp/ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe && \
    rm -rf /tmp/ffmpeg* && \
    apt-get purge -y -qq wget xz-utils > /dev/null 2>&1 && \
    apt-get autoremove -y -qq > /dev/null 2>&1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (as root, system-wide)
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --quiet --break-system-packages -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

WORKDIR /app
