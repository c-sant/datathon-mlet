from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from utils.config_loader import load_config

try:
    import yfinance as yf
except (ModuleNotFoundError, ImportError):
    yf = None

try:
    from newspaper import Article
except (ModuleNotFoundError, ImportError):
    Article = None


DEFAULT_URLS = []

# Fontes adicionais por ticker para aumentar cobertura de mercado.
DEFAULT_TICKERS = ["ITUB4", "PETR4", "VALE3", "BBAS3", "BBDC4"]


def _load_ingestion_ticker_settings():
    """Lê tickers e slugs do InfoMoney a partir da configuração de ingestão."""
    try:
        cfg = load_config()
        ingestion_cfg = (cfg.get("rag") or {}).get("ingestion") or {}
    except Exception:
        ingestion_cfg = {}

    tickers_cfg = ingestion_cfg.get("tickers") or []
    slugs_cfg = ingestion_cfg.get("infomoney_company_slugs") or {}

    tickers = [str(t).upper() for t in tickers_cfg if str(t).strip()]
    slugs = {
        str(k).upper(): str(v).strip().strip("/")
        for k, v in slugs_cfg.items()
        if str(k).strip() and str(v).strip()
    }
    return tickers, slugs


def _ticker_urls(tickers, infomoney_company_slugs=None):
    cfg_tickers, _ = _load_ingestion_ticker_settings()
    tickers = [str(t).upper() for t in (tickers or cfg_tickers or DEFAULT_TICKERS)]
    urls = []
    for ticker in tickers:
        ticker_lower = ticker.lower()
        urls.extend([
            f"https://statusinvest.com.br/acoes/{ticker_lower}",
            f"https://www.fundamentus.com.br/detalhes.php?papel={ticker.upper()}",
        ])
    return urls


def _extract_with_newspaper(url):
    """Extrai conteúdo com newspaper3k quando disponível."""
    if Article is None:
        return "", ""

    article = Article(url, language="pt")
    article.download()
    article.parse()
    return (article.title or "").strip(), (article.text or "").strip()


def _extract_with_html(url, timeout=12):
    """Fallback de extração quando newspaper3k falha ou retorna texto vazio."""
    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(" ", strip=True)

    blocks = []
    for node in soup.select("h1, h2, h3, p, li"):
        txt = node.get_text(" ", strip=True)
        if len(txt) >= 40:
            blocks.append(txt)

    text = "\n".join(blocks[:120]).strip()
    return title, text


def _doc_id(prefix, idx, url):
    host = urlparse(url).netloc.replace("www.", "").replace(".", "_")
    return f"{prefix}_{idx}_{host}"


def _load_yfinance_docs(tickers):
    """Gera documentos estruturados com dados fundamentalistas via yfinance."""
    if yf is None:
        return []

    cfg_tickers, _ = _load_ingestion_ticker_settings()
    tickers = [str(t).upper() for t in (tickers or cfg_tickers or DEFAULT_TICKERS)]

    docs = []
    for ticker in tickers:
        # B3 tickers need .SA suffix for Yahoo Finance
        yf_symbol = ticker if ticker.endswith(".SA") else f"{ticker}.SA"
        try:
            info = yf.Ticker(yf_symbol).info
            if not info or info.get("regularMarketPrice") is None:
                continue

            fields = [
                ("Empresa", info.get("longName") or info.get("shortName") or ticker),
                ("Setor", info.get("sector") or ""),
                ("Subsetor", info.get("industry") or ""),
                ("Preço atual", info.get("regularMarketPrice")),
                ("Variação dia (%)", info.get("regularMarketChangePercent")),
                ("Abertura", info.get("regularMarketOpen")),
                ("Máxima 52 semanas", info.get("fiftyTwoWeekHigh")),
                ("Mínima 52 semanas", info.get("fiftyTwoWeekLow")),
                ("Volume médio", info.get("averageVolume")),
                ("Market Cap", info.get("marketCap")),
                ("P/L", info.get("trailingPE")),
                ("P/VP", info.get("priceToBook")),
                ("EV/EBITDA", info.get("enterpriseToEbitda")),
                ("Dividend Yield (%)", info.get("dividendYield")),
                ("ROE (%)", info.get("returnOnEquity")),
                ("ROA (%)", info.get("returnOnAssets")),
                ("Margem líquida (%)", info.get("profitMargins")),
                ("Receita (TTM)", info.get("totalRevenue")),
                ("Lucro líquido (TTM)", info.get("netIncomeToCommon")),
                ("Dívida bruta", info.get("totalDebt")),
                ("Caixa", info.get("totalCash")),
                ("Beta", info.get("beta")),
                ("Recomendação analistas", info.get("recommendationKey")),
                ("Preço-alvo médio", info.get("targetMeanPrice")),
            ]

            lines = [f"Dados fundamentalistas de {ticker} ({yf_symbol}) — Yahoo Finance"]
            for label, val in fields:
                if val is not None and val != "":
                    if isinstance(val, float):
                        lines.append(f"{label}: {val:.4f}")
                    else:
                        lines.append(f"{label}: {val}")

            text = "\n".join(lines)
            if len(text) < 80:
                continue

            docs.append({
                "id": f"yfinance_{ticker.lower()}",
                "title": f"{ticker} — Indicadores fundamentalistas",
                "text": text,
                "source_url": f"https://finance.yahoo.com/quote/{yf_symbol}",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            print(f"Aviso: yfinance falhou para {ticker}: {e}")

    return docs


def load_news(urls=None, tickers=None, include_ticker_pages=True, infomoney_company_slugs=None):
    """
    Carrega notícias financeiras a partir de URLs.
    Se nenhuma lista for passada, usa as URLs default + páginas por ticker.
    """
    urls_to_fetch = list(urls or DEFAULT_URLS)
    if include_ticker_pages:
        urls_to_fetch.extend(_ticker_urls(tickers, infomoney_company_slugs))

    # Remove duplicatas preservando ordem.
    seen_urls = set()
    urls_to_fetch = [u for u in urls_to_fetch if not (u in seen_urls or seen_urls.add(u))]

    docs = []
    seen_payload = set()
    for i, url in enumerate(urls_to_fetch):
        title = ""
        text = ""

        try:
            title, text = _extract_with_newspaper(url)
        except Exception as e:
            print(f"Aviso: newspaper falhou em {url}: {e}")

        # Fallback quando o parser principal falha ou extrai pouco texto.
        if len(text) < 120:
            try:
                html_title, html_text = _extract_with_html(url)
                if html_title and not title:
                    title = html_title
                if len(html_text) > len(text):
                    text = html_text
            except Exception as e:
                print(f"Erro ao processar {url}: {e}")

        if len(text) < 120:
            continue

        payload_key = (title.strip().lower(), text[:240].strip().lower())
        if payload_key in seen_payload:
            continue
        seen_payload.add(payload_key)

        docs.append({
            "id": _doc_id("news", i, url),
            "title": title,
            "text": text,
            "source_url": url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        })

    # Complementa com dados fundamentalistas via yfinance (sem scraping HTML)
    if include_ticker_pages:
        docs.extend(_load_yfinance_docs(tickers))

    return docs
