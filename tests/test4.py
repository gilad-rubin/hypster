from classes import OpenAiDriver
from hypster import Options, lazy

OpenAiDriver = lazy(OpenAiDriver)

openai_model = Options(["gpt-4o", "gpt-4o-mini", "gpt-4"], default="gpt-4o")
llm_driver = OpenAiDriver(
    model=openai_model,
    max_tokens=Options({"low": 200, "medium": 500, "high": 1000}, default="medium")
)

tabular_driver = OpenAiDriver(model=openai_model, max_tokens=1500)