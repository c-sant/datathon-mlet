FROM python:3.10-slim

WORKDIR /app

# Copia arquivos
COPY . /app

# Instala BentoML e dependências
RUN pip install --no-cache-dir bentoml vllm torch transformers sentence-transformers faiss-cpu mlflow newspaper3k requests

# Expõe porta padrão do BentoML
EXPOSE 3000

# Comando para rodar o serviço
CMD ["bentoml", "serve", "app:svc", "--port", "3000"]