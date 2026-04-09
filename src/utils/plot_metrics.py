import json
import os
import pandas as pd
import matplotlib.pyplot as plt

METRICS_PATH = "reports/metrics.json"
OUTPUT_DIR = "reports"


def load_metrics():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    df = pd.DataFrame(metrics).T.reset_index()
    df.rename(columns={"index": "Modelo"}, inplace=True)

    # padroniza nomes
    df["Modelo"] = df["Modelo"].str.capitalize()

    return df


def plot_comparacao(df):
    plt.figure()
    df_plot = df.set_index("Modelo")[["mae", "rmse", "mape"]]
    df_plot.plot(kind="bar")
    plt.title("Comparação de Modelos")
    plt.ylabel("Valor")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparacao_modelos.png")
    plt.close()


def plot_ranking_mae(df):
    df_sorted = df.sort_values(by="mae")

    plt.figure()
    plt.bar(df_sorted["Modelo"], df_sorted["mae"])
    plt.title("Ranking por MAE (Erro Médio)")
    plt.ylabel("MAE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ranking_mae.png")
    plt.close()


def plot_mape(df):
    plt.figure()
    plt.bar(df["Modelo"], df["mape"])
    plt.title("Erro Percentual Médio (MAPE)")
    plt.ylabel("MAPE (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/mape.png")
    plt.close()


def main():
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError("metrics.json não encontrado. Rode o baseline primeiro.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_metrics()

    plot_comparacao(df)
    plot_ranking_mae(df)
    plot_mape(df)

    print("Gráficos gerados em /reports")


if __name__ == "__main__":
    main()