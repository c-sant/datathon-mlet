FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser

# Instala os pacotes mais pesados primeiro em camadas dedicadas.
# Assim o cache de camada do Docker é reaproveitado em qualquer rebuild
# que não altere as versões do torch/tensorflow.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install tensorflow>=2.15

COPY pyproject.toml .
COPY README.md .
COPY src/ src/
COPY data/ data/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install . --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

RUN mkdir -p data/raw models reports mlflow \
    && chown -R appuser:appuser /app \
    && git config --global --add safe.directory /app

USER appuser

CMD ["dvc", "repro"]