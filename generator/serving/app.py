import bentoml
from bentoml.io import JSON
from vllm import LLM, SamplingParams

# Inicializa o modelo via vLLM
llm = LLM(model="pierreguillou/gpt2-small-portuguese")

# Define parâmetros de geração
sampling_params = SamplingParams(temperature=0.7, max_tokens=200)

# Cria o serviço BentoML
svc = bentoml.Service(
    "acoes_generator",
    runners=[],
    description="Serviço de geração de texto em português usando vLLM",
)

# Define o endpoint /generate com entrada e saída JSON
@svc.api(
    input=JSON(
        example={
            "query": "Quais ações estão recomendadas para 2026?",
            "context": "Dados do retriever..."
        }
    ),
    output=JSON(
        example={
            "answer": "As ações recomendadas incluem diversificação em setores de tecnologia e energia renovável."
        }
    ),
)
def generate(input_json):
    query = input_json.get("query")
    context = input_json.get("context", "")
    prompt = f"Responda em português.\nPergunta: {query}\nContexto: {context}\nResposta:"
    
    outputs = llm.generate([prompt], sampling_params)
    return {"answer": outputs[0].outputs[0].text}