FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    FACEFUSION_DIR=/facefusion \
    FACEFUSION_PYTHON=/usr/local/bin/python \
    PORT=7860

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ARG FACEFUSION_VERSION=3.9.0
RUN git clone --depth 1 --branch ${FACEFUSION_VERSION} https://github.com/facefusion/facefusion.git /facefusion \
    && cd /facefusion \
    && python install.py default --skip-conda

COPY . /app

EXPOSE 7860
CMD ["python", "/app/app.py"]
