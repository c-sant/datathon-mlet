#!/bin/bash

echo "🧹 Limpando containers, imagens e volumes antigos..."

# Remove todos os containers (rodando e parados)
docker rm -f $(docker ps -aq) 2>/dev/null

# Remove todas as imagens não utilizadas
docker image prune -a -f

# Remove volumes não utilizados
docker volume prune -f

# Remove redes não utilizadas
docker network prune -f

echo "✅ Limpeza concluída!"