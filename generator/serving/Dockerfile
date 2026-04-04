FROM python:3.10-slim

WORKDIR /app

# Dependências básicas
RUN apt-get update && apt-get install -y build-essential python3-dev git

# Copia o projeto
COPY . /app

# Instala libs necessárias
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121 \
    vllm==0.4.1 bentoml transformers sentence-transformers faiss-cpu mlflow newspaper3k requests

EXPOSE 3000

# Força BentoML a rodar na porta 3000
CMD ["bentoml", "serve", "generator.serving.app:svc", "--port", "3000", "--host", "0.0.0.0"]
ENTRYPOINT ["bentoml", "serve", "generator.serving.app:svc", "--port", "3000", "--host", "0.0.0.0"]