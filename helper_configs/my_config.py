from hypster import HP


def my_config(hp: HP):
    chunking_strategy = hp.select(["paragraph", "semantic", "fixed"], default="paragraph")

    llm_model = hp.select(
        {"haiku": "claude-3-haiku-20240307", "sonnet": "claude-3-5-sonnet-20240620", "gpt-4o-mini": "gpt-4o-mini"},
        default="gpt-4o-mini",
    )

    llm_config = {"temperature": hp.number_input(0), "max_tokens": hp.number_input(64)}

    system_prompt = hp.text_input("You are a helpful assistant. Answer with one word only")
