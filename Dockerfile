FROM python:3.14-slim

# 日本語フォント(M3 レンダラの暫定字形ソース。字形バンク導入までのプレースホルダ)
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-ipaexfont-gothic \
        fonts-ipaexfont-mincho \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pillow

WORKDIR /work
COPY pipeline/ ./pipeline/

CMD ["python", "pipeline/run_phase0_bootstrap.py"]
