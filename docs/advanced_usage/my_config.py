from hypster import HP


def my_config(hp: HP):
    llm_model = hp.select(
        {
            "haiku": "claude-3-haiku-20240307",
            "sonnet": "claude-3-5-sonnet-20240620",
            "gpt-4o-mini": "gpt-4o-mini",
        },
        default="gpt-4o-mini",
    )
    llm_config = {
        "temperature": hp.number_input(0),
        "max_tokens": hp.number_input(64),
    }
