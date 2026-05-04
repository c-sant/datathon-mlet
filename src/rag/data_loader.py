from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

try:
    from newspaper import Article
except (ModuleNotFoundError, ImportError):
    Article = None

DEFAULT_URLS = [
    "https://www.seudinheiro.com/mercados",
    "https://einvestidor.estadao.com.br/mercado",
    "https://www.infomoney.com.br/mercados/",
]

# Fontes adicionais por ticker para aumentar cobertura de mercado.
DEFAULT_TICKERS = ["ITUB4", "PETR4", "VALE3", "BBAS3", "BBDC4"]


def _ticker_urls(tickers):
    tickers = [t.lower() for t in (tickers or DEFAULT_TICKERS)]
    urls = []
    for ticker in tickers:
        urls.extend([
            f"https://statusinvest.com.br/acoes/{ticker}",
            f"https://www.infomoney.com.br/cotacoes/b3/acao/{ticker}/",
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


def load_news(urls=None, tickers=None, include_ticker_pages=True):
    """
    Carrega notícias financeiras a partir de URLs.
    Se nenhuma lista for passada, usa as URLs default + páginas por ticker.
    """
    urls_to_fetch = list(urls or DEFAULT_URLS)
    if include_ticker_pages:
        urls_to_fetch.extend(_ticker_urls(tickers))

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

    return docs
