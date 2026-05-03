# Model Card — Modelos Preditivos de Preço de Fechamento

Model Card no padrão **Mitchell et al. (2019)** — *Model Cards for Model Reporting*, adaptado ao caso real do projeto **Datathon MLET / Grupo 05**.

Cobre os **três modelos de regressão** treinados pelo pipeline em [src/models/train.py](src/models/train.py), todos resolvendo a mesma tarefa com a mesma representação de entrada (justifica um Model Card consolidado em vez de três documentos separados):

| Variante | Framework | Artefato salvo | Função de treino |
|----------|-----------|----------------|------------------|
| `MLP_PyTorch` | PyTorch 2.x | `modelo_{ticker}_pytorch.pth` | [`treinar_pytorch`](src/models/train.py#L191-L212) |
| `MLPRegressor` | scikit-learn | `modelo_{ticker}_sklearn.joblib` | [train.py:363-378](src/models/train.py#L363-L378) |
| `LSTM` | Keras / TensorFlow | `modelo_{ticker}.keras` | [train.py:384-428](src/models/train.py#L384-L428) |
| `Baseline` (naïve persistence) | — | computado em runtime | [`criar_baseline_naive`](src/models/train.py#L229-L250), [src/models/baseline.py](src/models/baseline.py) |

> **Versão deste card.** v1.1 — 2026-05-03
> **Tracking.** MLflow em `mlflow/mlflow.db` (UI em `:5000` via [docker-compose.yaml:15-30](docker-compose.yaml#L15-L30)), experimento `previsao_acoes`.
> **Pipeline reprodutível.** [dvc.yaml](dvc.yaml) + [params.yaml](params.yaml).

---

## 1. Detalhes do modelo (Model Details)

### 1.1 Pessoa / organização desenvolvedora
**Grupo 05 — FIAP MLET, Fase Datathon.** Documento sob revisão do Encarregado LGPD a designar (ver [docs/LGPD_PLAN.md §8](docs/LGPD_PLAN.md#8-encarregado-dpo--art-41)).

### 1.2 Data
- **Treino mais recente.** Definida pelo `run_id` MLflow ativo (consultar `mlflow ui`).
- **Versão deste card.** v1.1 — 2026-05-03.

### 1.3 Tipo
- **Tarefa.** Regressão sobre série temporal **multivariada** (ver [src/features/feature_engineering.py](src/features/feature_engineering.py)).
- **Entrada.** Janela de `janela_dias × N_features` com features engenheiradas (lags, retornos, indicadores técnicos), escaladas por `MinMaxScaler` ([train.py:173-174](src/models/train.py#L173-L174)).
- **Saída.** Escalar — `target_next_close` previsto, normalizado; revertido pelo scaler para apresentação ([train.py:214-216](src/models/train.py#L214-L216)).

### 1.4 Arquitetura

#### MLP_PyTorch ([train.py:38-49](src/models/train.py#L38-L49))
```
Input(input_dim) → Linear(64) + ReLU → Linear(32) + ReLU → Linear(1)
```
- Otimizador: Adam, `lr` configurável em [config/model_config.yaml](config/model_config.yaml).
- Loss: MSE.
- Sem regularização explícita (sem dropout, sem weight decay).
- Treino implementado em [`treinar_pytorch`](src/models/train.py#L191-L212).

#### MLPRegressor (sklearn) ([train.py:363-378](src/models/train.py#L363-L378))
- `hidden_layer_sizes=(64, 32)`, `max_iter=500`, `random_state=42` ([train.py:364](src/models/train.py#L364)).
- Solver default `adam`, ativação default `relu`.
- Persistido via `joblib.dump` em [train.py:367](src/models/train.py#L367).

#### LSTM (Keras) ([train.py:384-416](src/models/train.py#L384-L416))
```
Input(janela_dias, n_features)
→ LSTM(50, return_sequences=True)
→ LSTM(50, return_sequences=False)
→ Dense(1)
```
- Otimizador: `adam`. Loss: MSE.
- `EarlyStopping` em [train.py:400](src/models/train.py#L400) com `restore_best_weights=True`.
- Disparado apenas com flag `--keras` ([train.py:384](src/models/train.py#L384)).

#### Baseline (naïve persistence)
`criar_baseline_naive` em [train.py:229-250](src/models/train.py#L229-L250) usa o `Close` do dia anterior como previsão. **Critério de comparação obrigatório.** Modelo só justifica deploy se vencer o baseline em MAE/RMSE/MAPE. Implementação alternativa em [src/models/baseline.py](src/models/baseline.py).

### 1.5 Hiperparâmetros
Centralizados em [config/model_config.yaml](config/model_config.yaml) e [params.yaml](params.yaml), carregados por [src/utils/config_loader.py](src/utils/config_loader.py):
- `pytorch_mlp.epochs`, `pytorch_mlp.optimizer.learning_rate`
- `sklearn_mlp.hidden_layer_sizes`, `sklearn_mlp.max_iter`, `sklearn_mlp.random_state`
- `keras_lstm.epochs`, `keras_lstm.batch_size`, `keras_lstm.early_stopping.patience`
- `data.janela_dias`, `data.ticker`, `data.start_date`, `data.end_date`

### 1.6 Versionamento
- **Modelo.** Salvo em `models/modelo_{ticker}_pytorch.pth` ([train.py:342](src/models/train.py#L342)), `models/modelo_{ticker}_sklearn.joblib` ([train.py:363](src/models/train.py#L363)), `models/modelo_{ticker}.keras` ([train.py:416](src/models/train.py#L416)).
- **Run.** `run_id` exposto via `print(f"run_id={run.info.run_id}")` ([train.py:435](src/models/train.py#L435)).
- **Dados.** [dvc.yaml](dvc.yaml) + `dvc.lock` para reprodutibilidade.
- **Tags MLflow.** `model_type`, `framework`, `owner`, `phase`, `problem_type`, `dataset_type` definidas em `log_tags_padronizadas` ([train.py:253-259](src/models/train.py#L253-L259)).

### 1.7 Licença e contato
- **Licença.** Definir (sugerido: MIT ou Apache 2.0 — dependendo da política da FIAP).
- **Contato.** Encarregado LGPD a designar — ver [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md).

---

## 2. Uso pretendido (Intended Use)

### 2.1 Casos de uso primários
- **Demonstração acadêmica** de pipeline MLOps fim a fim (treino + tracking + serving).
- **Comparação metodológica** entre frameworks (PyTorch / Sklearn / Keras) sobre série univariada.
- **Estudo de caso** de previsão de série temporal financeira para fins didáticos.

### 2.2 Usuários primários
Estudantes, professores e avaliadores do programa MLET. Não há provisão para uso por investidores leigos ou por sistemas automatizados de execução.

### 2.3 Fora do escopo (Out-of-scope)
- **Trading automatizado** ou execução de ordens.
- **Recomendação a clientes finais** (atividade regulada — Resolução CVM 39/2021).
- **Decisão de crédito ou seguro** com base na previsão.
- **Ativos de baixa liquidez** (small caps com histórico curto).
- **Janelas com poucos pontos** — [train.py:148-149](src/models/train.py#L148-L149) lança `ValueError` mas a checagem mínima de qualidade é insuficiente para uso em produção.

---

## 3. Fatores (Factors)

Fatores que afetam significativamente a performance e devem ser considerados em qualquer avaliação.

### 3.1 Fatores relevantes
| Fator | Por que importa | Como avaliar |
|-------|-----------------|--------------|
| **Liquidez do ativo** | Ativos pouco líquidos têm gaps em `Close`, distorcendo lags | Estratificar MAPE por `volume_médio_diário` |
| **Setor econômico** | Setores reagem diferentemente a regimes macro | Estratificar MAPE por setor B3 |
| **Tamanho do histórico** | IPOs recentes têm pouco contexto para janela `t-N` | Estratificar por `dias_desde_ipo` |
| **Regime macroeconômico** | Modelo treinado em Selic baixa pode falhar em Selic alta | PSI sobre janelas treino vs produção |
| **Eventos exógenos** | Splits, dividendos, fusões, *halts* | Avaliação manual; modelo não trata |
| **Janela escolhida** | Janela curta perde tendência; janela longa dilui sinal recente | Sensibilidade da métrica a `janela_dias` |

### 3.2 Avaliação atual por fator
**Não realizada.** O pipeline atual só reporta métricas agregadas. O plano em [docs/EXPLAINABILITY_FAIRNESS.md §6 Sprint 3](docs/EXPLAINABILITY_FAIRNESS.md#6-plano-de-implementação) detalha a instrumentação por subgrupo.

---

## 4. Métricas (Metrics)

### 4.1 Métricas de erro (já instrumentadas)
Calculadas em `avaliar_regressao` ([train.py:214-227](src/models/train.py#L214-L227)) e logadas no MLflow para baseline ([train.py:331-333](src/models/train.py#L331-L333)) e cada modelo.

| Métrica | Fórmula | Limiar de aprovação |
|---------|---------|---------------------|
| MAE | `mean(|y - ŷ|)` | < MAE do baseline |
| RMSE | `sqrt(mean((y - ŷ)²))` | < RMSE do baseline |
| MAPE | `mean(|(y - ŷ) / y|) × 100` | < MAPE do baseline (e idealmente < 5%) |

> Avaliação aplicada em escala original (após `inverse_transform` em [train.py:215-216](src/models/train.py#L215-L216)).

### 4.2 Métricas planejadas
| Métrica | Para que serve |
|---------|----------------|
| **Skill score vs baseline** | `1 - MSE_modelo / MSE_baseline` — quão melhor que persistence |
| **MAPE por subgrupo** | Detectar performance gap (fairness) |
| **PSI** | Detectar drift entre treino e produção |
| **Confidence interval (MC Dropout)** | Banda de incerteza ao redor da previsão pontual |
| **Diretional accuracy** | Frequência com que o modelo acerta o sinal de variação |

### 4.3 Limiar de "aprovação" para deploy
Modelo é considerado válido para uso (didático) quando:
1. Vence o baseline em **as três** métricas (MAE, RMSE, MAPE).
2. MAPE < 5% no teste.
3. PSI treino↔teste < 0.1.
4. Performance gap entre maior e menor subgrupo < 1.5×.

Qualquer modelo abaixo destes limiares **não** deve ser anunciado como "previsor confiável".

---

## 5. Dados de treino (Training Data)

### 5.1 Fonte
CSV de features tratadas em `data/raw/stock_features.csv`, gerado automaticamente por [`ensure_features_dataset`](src/models/train.py#L52-L82) a partir de `data/raw/stock_data.csv` via [src/features/feature_engineering.py](src/features/feature_engineering.py).

### 5.2 Pré-processamento
1. Geração de features (lags, retornos, indicadores técnicos) por `feature_engineering.py` — documentada em [src/features/ReadmeFeatureEngineering.md](src/features/ReadmeFeatureEngineering.md).
2. Seleção de colunas de feature por `get_feature_columns` ([train.py:83](src/models/train.py#L83)).
3. Carga via `carregar_dados_csv` ([train.py:99](src/models/train.py#L99)).
4. Normalização do target via `MinMaxScaler` em `[0, 1]` ([train.py:173-174](src/models/train.py#L173-L174)).
5. Construção de janelas deslizantes 3D `janela_dias × n_features` em `preparar_series_features` ([train.py:154-188](src/models/train.py#L154-L188)).

### 5.3 Split
**Temporal** 80/20 — primeiros 80% no treino, últimos 20% no teste; tipo registrado como tag MLflow `split_type=temporal_80_20` ([train.py:317](src/models/train.py#L317)). Crítico para séries temporais; **não usar** split aleatório (data leakage).

### 5.4 Características conhecidas
- **Multivariada.** Features engenheiradas via `feature_engineering.py` — `Close`, lags, retornos e indicadores técnicos.
- **Sem features exógenas externas.** Macroeconomia, calendário, notícias e eventos não são incorporados ao input do modelo (entram apenas via RAG).
- **Sem augmentation.** Não há *bootstrapping* nem síntese de exemplos.

### 5.5 Distribuição
**Não documentada explicitamente.** Recomenda-se anexar ao card, por release:
- Período (`start_date` → `end_date`).
- Número de pontos antes e depois de drop NaN.
- Estatísticas descritivas de `Close` (média, mediana, std, min, max, *skew*, *kurtosis*).
- Gráfico da série completa marcando o split.

---

## 6. Dados de avaliação (Evaluation Data)

### 6.1 Conjunto de teste
Os últimos 20% da série, **mesma ordem temporal** do conjunto de treino. Logado como `split_type=temporal_80_20` em [train.py:317](src/models/train.py#L317).

### 6.2 Limitação importante
Como o conjunto de teste é uma janela contígua, ele representa **um único regime econômico**. Métricas de teste **não generalizam** para regimes diferentes. É necessário:
- Validação *walk-forward* (múltiplas janelas de teste deslocadas no tempo).
- Avaliação em períodos de *stress* (crise 2008, COVID-2020, ciclo Selic alta 2022-2024).

### 6.3 Conjunto de validação (LSTM apenas)
O Keras usa `validation_data` no `model.fit` em [train.py:412](src/models/train.py#L412) com o próprio conjunto de teste, o que **vaza informação para o `EarlyStopping`** e infla a métrica reportada.

**Ação.** Refatorar para split treino / val / teste explícito (ex.: 70/10/20).

---

## 7. Análises quantitativas (Quantitative Analyses)

### 7.1 Performance agregada
Os valores numéricos são reportados em runtime via MLflow. Para inspecionar:

```bash
docker compose up mlflow
# UI em http://localhost:5000
```

Filtrar por experimento `previsao_acoes` e comparar `mae_baseline` / `mae_pytorch` / `mae_sklearn` / `mae_keras` (e equivalentes para RMSE, MAPE). As mesmas métricas são também ingeridas no índice RAG via [src/rag/mlflow_loader.py](src/rag/mlflow_loader.py) e expostas pelo endpoint `POST /ingest_mlflow`.

### 7.2 Performance desagregada
**Não disponível.** A entregar conforme [docs/EXPLAINABILITY_FAIRNESS.md §5.3](docs/EXPLAINABILITY_FAIRNESS.md#53-métricas-a-instrumentar):
- MAPE por setor B3.
- MAPE por faixa de liquidez.
- MAPE por regime macro.

### 7.3 Comparação contra baseline
Reportada no log:
```
[Baseline] MAE=... | RMSE=... | MAPE=...
[PyTorch]  MAE=... | RMSE=... | MAPE=...
[Scikit]   MAE=... | RMSE=... | MAPE=...
[Keras]    MAE=... | RMSE=... | MAPE=...
```

Em séries de fechamento diário com baixa volatilidade, é **comum** o baseline persistence ser muito difícil de bater — esse é um sinal importante de que o ganho preditivo é marginal.

---

## 8. Considerações éticas (Ethical Considerations)

### 8.1 Risco principal: confundir previsão estatística com recomendação
O modelo produz um número (próximo fechamento). Sem disclaimer e contexto, um leitor pode interpretar como "compre/venda". Risco material.

**Mitigação.** Disclaimer obrigatório (ver [docs/RED_TEAM_REPORT.md §RT-02](docs/RED_TEAM_REPORT.md#rt-02--data-poisoning-via-ingest-ingest_mlflow-e-fetch_news)) e separação clara entre `prediction` (número), `confidence_interval`, e ausência de qualquer indicação de ação a tomar.

### 8.2 Risco de uso em populações para as quais o modelo não foi avaliado
Modelo treinado em `--ticker` específico **não generaliza** para outros tickers. Avaliar antes de reutilizar.

### 8.3 Dados sensíveis
**Não há dados pessoais** nos dados de treino (apenas séries de preço). O modelo final em si não é dado pessoal.

### 8.4 Decisões automatizadas (LGPD Art. 20)
Caso a saída do modelo venha a ser usada para **decidir algo sobre uma pessoa** (ex.: aprovar limite, sugerir investimento personalizado), o titular tem direito à revisão humana. Implementação prevista em [docs/LGPD_PLAN.md §5](docs/LGPD_PLAN.md#5-direitos-dos-titulares-art-18--como-atender).

### 8.5 Não discriminação (LGPD Art. 6º, IX)
Aplicar fairness por subgrupo (R1 em [docs/EXPLAINABILITY_FAIRNESS.md](docs/EXPLAINABILITY_FAIRNESS.md#52-riscos-concretos-no-projeto)) é parte do compromisso de não discriminação. Modelo que falha sistematicamente em ativos de menor liquidez tende a marginalizá-los.

---

## 9. Caveats e recomendações (Caveats and Recommendations)

### 9.1 O que este modelo **não** faz
- ❌ Não prevê retorno acumulado, volatilidade ou risco de drawdown.
- ❌ Não considera notícias, fundamentos ou macroeconomia.
- ❌ Não dá intervalo de confiança (planejado).
- ❌ Não detecta sozinho que a distribuição mudou (planejado via PSI).
- ❌ Não recomenda compra/venda — é apenas um *estimador pontual* do próximo fechamento.

### 9.2 O que fazer antes de um próximo deploy
1. Adicionar **validação walk-forward** com pelo menos 3 janelas.
2. Corrigir o vazamento de teste no LSTM (split treino/val/teste).
3. Instrumentar **performance desagregada** por subgrupo (R1).
4. Adicionar **CI via MC Dropout** ou bootstrap.
5. Adicionar **explicabilidade** (SHAP / Integrated Gradients) — Sprint 2 em [docs/EXPLAINABILITY_FAIRNESS.md §6](docs/EXPLAINABILITY_FAIRNESS.md#6-plano-de-implementação).
6. Comparar com baselines mais fortes (ARIMA, Prophet, naive sazonal).

### 9.3 Quando descontinuar
Conforme critérios de sunset em [docs/SYSTEM_CARD.md §12.3](docs/SYSTEM_CARD.md#123-critérios-para-retirar-de-uso-sunset):
- PSI > 0.5 sustentado por 7 dias.
- MAPE em produção > 2× MAPE de teste por 7 dias.
- Vazamento de teste no LSTM **não corrigido** até a próxima release.

---

## 10. Histórico de versões

| Versão | Data | Mudança | Observações |
|--------|------|---------|-------------|
| 1.0 | 2026-04-28 | Versão inicial do Model Card consolidado | Cobre PyTorch, Sklearn, Keras (univariado) |
| 1.1 | 2026-05-03 | Atualização para pipeline multivariado | Reflete `feature_engineering.py`, novo layout de `train.py`, integração com `/ingest_mlflow` |

---

## Documentos relacionados

- [docs/SYSTEM_CARD.md](docs/SYSTEM_CARD.md) — System Card (visão holística)
- [docs/EXPLAINABILITY_FAIRNESS.md](docs/EXPLAINABILITY_FAIRNESS.md) — Explicabilidade e fairness
- [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md) — Conformidade LGPD
- [docs/OWASP.md](docs/OWASP.md) / [docs/RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md) — Segurança
- [src/features/ReadmeFeatureEngineering.md](src/features/ReadmeFeatureEngineering.md) / [src/models/README_TRAIN_BASELINE.md](src/models/README_TRAIN_BASELINE.md) — Detalhes do pipeline
- Mitchell et al., *[Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)*, 2019
