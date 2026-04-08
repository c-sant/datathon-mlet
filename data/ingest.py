import argparse
import os
import pandas as pd
import yfinance as yf


def main(args):
    print(f"Baixando dados: {args.ticker}")

    df = yf.download(args.ticker, start=args.start, end=args.end, progress=False)

    if df.empty:
        raise ValueError("Nenhum dado retornado")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output)

    print(f"Dados salvos em: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()
    main(args)
