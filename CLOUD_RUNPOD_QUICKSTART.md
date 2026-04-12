# Quickstart: vLLM Quantizado na Nuvem (RunPod) + BentoML Local

Este fluxo valida o requisito `LLM servido via API com quantizacao aplicada` sem depender de GPU local.

## Arquitetura

```
Cliente -> BentoML local (porta 3004) -> vLLM RunPod (GPU NVIDIA) -> BentoML -> Cliente
```

- vLLM quantizado roda no RunPod com GPU NVIDIA
- BentoML roda localmente, encaminha requisicoes para o vLLM remoto via `VLLM_BASE_URL`

## 1) Criar pod no RunPod

### Configuracao validada

- Imagem: `vllm/vllm-openai:latest`
- GPU: RTX 2000 Ada (ou superior)
- Porta HTTP: `8000`
- Storage: network disk, 40 GB persistente

### Comando do pod (campo "Docker Command" no RunPod UI)

Cole exatamente os argumentos abaixo (formato args-only, sem `python -m`):

```
--model Qwen/Qwen2.5-0.5B-Instruct-AWQ
--served-model-name qwen2.5-0.5b-awq
--quantization awq
--dtype half
--host 0.0.0.0
--port 8000
--max-model-len 1024
--gpu-memory-utilization 0.75
--enforce-eager
--api-key <sua-api-key>
```

Gere uma api-key segura antes de criar o pod:

```powershell
[System.Convert]::ToBase64String((1..32 | ForEach-Object { Get-Random -Max 256 }) -as [byte[]])
```

### Variavel de ambiente opcional

Se o modelo exigir autenticacao no Hugging Face:

- `HF_TOKEN=<seu_token_hf>`

## 2) Validar vLLM remoto

Apos o pod mostrar status `Running` e o modelo terminar de carregar (~2-3 min), valide:

```powershell
$VLLM_URL = "https://SEU_ENDPOINT_RUNPOD"
$VLLM_KEY = "sua-api-key"

Invoke-WebRequest -Uri "$VLLM_URL/health" `
  -Headers @{ "Authorization" = "Bearer $VLLM_KEY" } `
  -Method Get -UseBasicParsing

Invoke-WebRequest -Uri "$VLLM_URL/v1/models" `
  -Headers @{ "Authorization" = "Bearer $VLLM_KEY" } `
  -Method Get -UseBasicParsing
```

Ambos devem retornar status 200.

## 3) Subir BentoML local apontando para RunPod

```powershell
cd "C:\Users\cabri\Documents\Fiap\challenge5\datathon-grupo-05"
$env:VLLM_BASE_URL = "https://SEU_ENDPOINT_RUNPOD"
$env:VLLM_MODEL = "qwen2.5-0.5b-awq"
$env:VLLM_API_KEY = "sua-api-key"

docker compose -f docker/docker-compose.bento.remote.yml down
docker compose -f docker/docker-compose.bento.remote.yml up -d --build
```

Confirme que as variaveis entraram no container:

```powershell
docker compose -f docker/docker-compose.bento.remote.yml exec bentoml_remote sh -lc "printenv | grep VLLM"
```

## 4) Teste da API BentoML local

Use o bloco abaixo para evitar problema de encoding UTF-8 no PowerShell 5.1:

```powershell
$payload = @{
  query   = "Resuma o cenario de investimento para 2026"
  context = "Juros altos favorecem renda fixa e selecao criteriosa de acoes."
} | ConvertTo-Json -Compress

$bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)

$resp = Invoke-WebRequest -Method Post `
  -Uri "http://localhost:3004/generate" `
  -ContentType "application/json; charset=utf-8" `
  -Body $bytes

$reader = New-Object System.IO.StreamReader($resp.RawContentStream, [System.Text.Encoding]::UTF8)
($reader.ReadToEnd() | ConvertFrom-Json).answer
```

## 5) Boas praticas de custo no RunPod

- Use pods spot/community quando disponivel
- Desligue o pod quando nao estiver usando (pausar gasto)
- Modelo 0.5B AWQ usa menos de 2 GB VRAM - cabe em qualquer GPU disponivel
- Evite rebuild do BentoML desnecessario (use `up -d` sem `--build` apos primeiro build)

## 6) Observacoes tecnicas

- O vLLM e quem aplica quantizacao AWQ nos pesos do modelo (`--quantization awq`)
- BentoML apenas expoe a API de negocio e encaminha para o vLLM
- `VLLM_API_KEY` e opcional: se nao configurado no pod, remova `--api-key` e `VLLM_API_KEY`
- Este fluxo nao altera o stack CPU local estavel (`docker/docker-compose.yml`)
