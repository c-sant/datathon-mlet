---

## 🔹 Microsoft MLOps Maturity Model – Experiment Management

A dimensão *Experiment Management* avalia como os experimentos de machine learning são organizados, rastreados e comparados.  
Nosso projeto já cobre de forma sólida os níveis 2 e 3.

| Nível | Descrição | O que temos implementado |
|-------|-----------|---------------------------|
| **Stage 1 – Inicial** | Experimentos ad hoc, sem rastreabilidade. | ❌ Não aplicável, já superado. |
| **Stage 2 – Básico** | Registro manual de métricas e parâmetros, pouca padronização. | ✅ Runs separados por framework.<br>✅ Métricas (MAE, RMSE, MAPE) logadas.<br>✅ Parâmetros registrados (ticker, janela, epochs, batch_size, patience). |
| **Stage 3 – Intermediário** | Uso de ferramenta de tracking (MLflow), runs padronizados, comparabilidade sistemática. | ✅ MLflow padronizado.<br>✅ Artefatos registrados (modelos, CSV, gráficos).<br>✅ Benchmark automatizado para comparação entre frameworks. |
| **Stage 4 – Avançado** | Automação de experimentos, versionamento de datasets, integração com pipelines CI/CD. | ⚠️ Ainda não implementado. Planejado para evolução futura. |
| **Stage 5 – Otimizado** | Dashboards interativos, governança, auditoria e reprodutibilidade total. | ⚠️ Ainda não implementado. Planejado para evolução futura. |

---

📌 **Conclusão:**  
Estamos **bem estruturados até os níveis 2 e 3**, garantindo rastreabilidade, reprodutibilidade e comparabilidade dos experimentos.