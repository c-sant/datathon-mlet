# 🧪 Testes Manuais com Pytest (Python 3.13)

Este documento descreve como executar os testes automatizados do projeto manualmente, utilizando **Python 3.13** e também via **Docker**.

---

## ⚠️ Observação sobre Python 3.13

O projeto está sendo executado com:

```bash
python --version

Exemplo:

Python 3.13.x

🔴 Atenção:
Algumas bibliotecas podem ter suporte parcial no Python 3.13:

TensorFlow
PyTorch (dependendo da versão)
MLflow

👉 Apesar disso, os testes funcionam normalmente utilizando CPU.

📁 Estrutura de Testes

tests/
├── conftest.py
├── test_api.py
├── test_ingest.py
├── test_baseline.py
├── test_train.py
├── test_plot_metrics.py
⚙️ Pré-requisitos

# Execução local (sem Docker)
Python 3.13
Ambiente virtual ativo (recomendado)

# Execução via Docker
Docker Desktop instalado
Docker em execução
(WSL integrado, caso esteja usando Linux via Windows)

🧪 Execução LOCAL (sem Docker)

Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate
Instalar dependências
pip install -e ".[test]"

# Rodar testes
pytest

🐳 Execução via Docker (RECOMENDADO)

O projeto possui ambiente isolado para testes utilizando Dockerfile.test.

🔧 Build da imagem de teste

docker compose build test

ou (modo manual):

docker build -f Dockerfile.test -t datathon-mlet-test .

▶️ Rodar os testes

Usando Docker Compose
docker compose run --rm test
Usando Docker direto
docker run --rm datathon-mlet-test

📊 Saída esperada
collected XX items

tests/test_api.py::test_health_returns_200 PASSED
tests/test_ingest.py::test_ingest_success PASSED
tests/test_baseline.py::test_avaliar_modelo PASSED
...

🧠 Observações importantes

✔ 1. Imports funcionando automaticamente

O projeto utiliza configuração no pyproject.toml:

pythonpath = ["."]

👉 Isso permite que o pytest reconheça:

src/
data/
app/

✔ 2. Fixtures (conftest.py)

Os testes utilizam dados sintéticos:

Simulação de dados de ações
CSVs temporários
Cenários inválidos

✔ 3. Testes com Mock

Uso de monkeypatch para:

Simular yfinance.download
Evitar chamadas externas
Garantir execução offline

✔ 4. Execução em CPU

Mensagem esperada:

CUDA initialization...
GPU will not be used

✔ 5. Docker vs Local

Execução	Quando usar
Local (pytest)	Desenvolvimento rápido
Docker	Ambiente isolado / CI / padrão

✔ 6. Docker Desktop

Para execução via Docker:

É obrigatório ter o Docker Desktop instalado
O serviço deve estar em execução
No Windows com WSL, é necessário ativar a integração

🧪 Cobertura de Testes
pytest --cov=src --cov=data

Ou:

pytest --cov=src --cov=data --cov-report=term-missing

🚀 Boas práticas aplicadas
✔ Testes isolados
✔ Dados sintéticos
✔ Mock de APIs externas
✔ Testes de ML (treino + inferência)
✔ Testes de geração de gráficos
✔ Validação de erros

📌 Execução resumida

# Local
pip install -e ".[test]"
pytest

# Docker
docker compose build test
docker compose run --rm test