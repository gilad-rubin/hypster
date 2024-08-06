from classes import OpenAiDriver
from hypster import Options, lazy

OpenAiDriver = lazy(OpenAiDriver)

# Example 2: Complex data-types (Class, Functions) - Part 2
llm_driver = OpenAiDriver(
    model=Options(["gpt-4o", "gpt-4o-mini", "gpt-4"], default="gpt-4o"),
    max_tokens=Options({"low": 200, "medium": 500, "high": 1000}, default="medium")
)