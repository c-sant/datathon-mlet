import bentoml
from bentoml.io import JSON
from vllm import LLM, SamplingParams

# Inicializa o modelo via vLLM (substitua pelo modelo que você já tem de ações)
llm = LLM(model="pierreguillou/gpt2-small-portuguese")

# Define parâmetros de geração
sampling_params = SamplingParams(temperature=0.7, max_tokens=200)

svc = bentoml.Service("acoes_generator", runners=[])

@svc.api(input=JSON(), output=JSON())
def generate(input_json):
    query = input_json.get("query")
    context = input_json.get("context", "")
    prompt = f"Responda em português.\nPergunta: {query}\nContexto: {context}\nResposta:"
    
    outputs = llm.generate([prompt], sampling_params)
    return {"answer": outputs[0].outputs[0].text}