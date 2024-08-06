from hypster import Options, lazy
from .classes import OpenAiDriver

OpenAiDriver = lazy(OpenAiDriver)

model = Options(["gpt-4o", "gpt-4o-mini", "gpt-4"], default="gpt-4o")
llm_driver = OpenAiDriver(
    model=model,
    max_tokens=Options({"low": 200, "medium": 500, "high": 1000}, default="medium")
)