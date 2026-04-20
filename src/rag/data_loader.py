from newspaper import Article

DEFAULT_URLS = [
    "https://www.seudinheiro.com/mercados",
    "https://einvestidor.estadao.com.br/mercado",
    "https://www.infomoney.com.br/mercados/",
]


def load_news(urls=None):
    """
    Carrega notícias financeiras a partir de URLs.
    Se nenhuma lista for passada, usa as URLs default.
    """
    if urls is None:
        urls = DEFAULT_URLS

    docs = []
    for i, url in enumerate(urls):
        try:
            article = Article(url, language="pt")
            article.download()
            article.parse()
            docs.append({"id": f"news_{i}", "title": article.title, "text": article.text})
        except Exception as e:
            print(f"Erro ao processar {url}: {e}")
    return docs
