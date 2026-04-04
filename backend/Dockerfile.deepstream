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

# Install yt-dlp + deno (precompiled binaries, latest release)
# Required for YouTube Live URL resolution (M6).
# deno is needed by yt-dlp for YouTube JavaScript extraction.
RUN apt-get update -qq && \
    apt-get install -y -qq curl unzip > /dev/null 2>&1 && \
    curl -sL https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp \
      -o /usr/local/bin/yt-dlp && \
    chmod +x /usr/local/bin/yt-dlp && \
    curl -sL https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip \
      -o /tmp/deno.zip && \
    unzip -o -q /tmp/deno.zip -d /usr/local/bin/ && \
    chmod +x /usr/local/bin/deno && \
    rm /tmp/deno.zip && \
    apt-get purge -y -qq unzip > /dev/null 2>&1 && \
    apt-get autoremove -y -qq > /dev/null 2>&1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (as root, system-wide)
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --quiet --break-system-packages -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

WORKDIR /app
