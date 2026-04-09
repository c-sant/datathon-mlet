FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Git fix
RUN git config --global --add safe.directory /app

# cria usuário
RUN useradd -m appuser

# COPY otimizado
COPY pyproject.toml .
COPY README.md .
COPY src/ src/

RUN pip install --upgrade pip && pip install .

COPY . .

# troca usuário
USER appuser

CMD ["dvc", "repro"]