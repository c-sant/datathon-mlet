import importlib
import sys
import types


def _clear_module(name):
    sys.modules.pop(name, None)


def _stub_embedding_dependencies(monkeypatch, sentence_transformer_impl):
    fake_st_module = types.ModuleType("sentence_transformers")
    fake_st_module.SentenceTransformer = sentence_transformer_impl
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_module)

    fake_faiss = types.ModuleType("faiss")

    class _FakeIndexFlatL2:
        def __init__(self, dim):
            self.dim = dim

        def add(self, embeddings):
            return None

    fake_faiss.IndexFlatL2 = _FakeIndexFlatL2
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)

    fake_mlflow = types.ModuleType("mlflow")

    class _DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_mlflow.start_run = lambda *args, **kwargs: _DummyRun()
    fake_mlflow.log_param = lambda *args, **kwargs: None
    fake_mlflow.log_artifact = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    fake_loader = types.ModuleType("rag.data_loader")
    fake_loader.load_news = lambda: []
    monkeypatch.setitem(sys.modules, "rag.data_loader", fake_loader)


def _stub_generator_dependencies(monkeypatch, pipeline_impl):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = pipeline_impl
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_embedding_reads_hugging_face_hub_token_alias(monkeypatch):
    calls = []

    def fake_sentence_transformer(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return object()

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "token-from-alias")

    _stub_embedding_dependencies(monkeypatch, fake_sentence_transformer)
    _clear_module("rag.embedding")

    embedding = importlib.import_module("rag.embedding")

    assert embedding.HF_TOKEN == "token-from-alias"
    assert calls
    assert calls[0][1]["token"] == "token-from-alias"


def test_embedding_falls_back_to_use_auth_token(monkeypatch):
    calls = []

    def fake_sentence_transformer(model_name, **kwargs):
        calls.append((model_name, kwargs))
        if "token" in kwargs:
            raise TypeError("token unsupported")
        return object()

    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    _stub_embedding_dependencies(monkeypatch, fake_sentence_transformer)
    _clear_module("rag.embedding")

    importlib.import_module("rag.embedding")

    assert len(calls) >= 2
    assert "token" in calls[0][1]
    assert calls[1][1]["use_auth_token"] == "hf-token"


def test_generator_reads_legacy_huggingfacehub_api_token(monkeypatch):
    calls = []

    def fake_pipeline(task, **kwargs):
        calls.append((task, kwargs))
        return object()

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "legacy-token")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    _stub_generator_dependencies(monkeypatch, fake_pipeline)
    _clear_module("rag.generator")

    generator = importlib.import_module("rag.generator")
    generator._build_text_generator("distilgpt2")

    assert generator.HF_TOKEN == "legacy-token"
    assert calls
    assert calls[0][0] == "text-generation"
    assert calls[0][1]["token"] == "legacy-token"


def test_generator_falls_back_to_use_auth_token(monkeypatch):
    calls = []

    def fake_pipeline(task, **kwargs):
        calls.append((task, kwargs))
        if "token" in kwargs:
            raise TypeError("token unsupported")
        return object()

    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    _stub_generator_dependencies(monkeypatch, fake_pipeline)
    _clear_module("rag.generator")

    generator = importlib.import_module("rag.generator")
    generator._build_text_generator("gpt2")

    assert len(calls) == 2
    assert "token" in calls[0][1]
    assert calls[1][1]["use_auth_token"] == "hf-token"
